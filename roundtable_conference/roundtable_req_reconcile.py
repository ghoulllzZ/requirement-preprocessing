# roundtable_req_reconcile.py
# -*- coding: utf-8 -*-
"""
Per-requirement roundtable (RECONCILE-style) for requirement quality auditing.

Pipeline:
A) Initial scoring per requirement by multiple LLMs:
   - scores (U/A/C/V), confidences (U/A/C/V), rationales, rewrites, issues (taxonomy=U/A/C/V)
B) Fixed rater weights:
   - Option 1: read weights.csv
   - Option 2: compute from llm_rating_report.xlsx (analyze_llm_likert_scores output) + human labels:
       model_labels.csv (model-level) + hotcell_labels.csv (hot cells only)
C) Roundtable:
   - Discuss TOP-K issue candidates (item, type) ranked by influence = w_r * recalibrated_conf
   - In each round, each rater updates scores/confidences/issues for provided items
   - Stop if any: score convergence OR issue convergence OR max rounds
D) Final output:
   - item + issue_type(U/A/C/V) + evidence + rewrite (issue aggregated score >= threshold)
   - Excel report

OpenAI-compatible endpoint: POST {base_url}/v1/chat/completions
"""

import os
import re
import json
import ast
import math
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# =========================
# 0) Constants / Taxonomy
# =========================
DIMENSIONS = ["Unambiguous", "Understandable", "Correctness", "Verifiable"]
DIM_SHORT = {
    "Unambiguous": "U",
    "Understandable": "A",
    "Correctness": "C",
    "Verifiable": "V",
}
SHORT_DIM = {v: k for k, v in DIM_SHORT.items()}

DEFAULT_TOPK_ISSUES = 10
DEFAULT_MAX_ROUNDS = 2

# Issue aggregation threshold: Theta = 0.6 * sum(weights); if weights normalized sum=1 -> 0.6
DEFAULT_THETA_RATIO = 0.6

# Convergence
DEFAULT_EPS_SCORE = 0.25  # score convergence epsilon
DEFAULT_TAU_JACCARD = 0.9  # issue-set convergence

# Confidence recalibration (RECONCILE-style)
def recalibrate_conf(p: float) -> float:
    p = float(p)
    if p >= 1.0:
        return 1.0
    if p >= 0.9:
        return 0.8
    if p >= 0.8:
        return 0.5
    if p > 0.6:
        return 0.3
    return 0.1


# Hot-cell criteria (agree with your default)
HOTCELL_RULES = {
    "range_ge": 2.0,
    "iqr_ge": 1.0,
    "abs_delta_ge": 2.0,
}

# Weight computation hyperparams (only used if not providing weights.csv)
LAMBDA_BAD_GLOBAL = 1.2
LAMBDA_BAD_HOT = 1.0
LAMBDA_GOOD_HOT = 0.6

# For OutlierIndex built from analyze report metrics (min-max normalized, then weighted sum)
OUTLIER_METRICS_WEIGHTS = {
    "avg_distance_to_others": 0.35,
    "contribution(G-G_minus)": 0.35,
    "bias_abs": 0.15,
    "mean_abs_delta": 0.15,
}

# Network / retry
TEMPERATURE_SCORE = 0.2
TEMPERATURE_DISCUSS = 0.2
MAX_RETRIES = 2
RETRY_SLEEP = 2.0


# =========================
# 1) Providers
# =========================
class LLMProvider:
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        raise NotImplementedError


class OpenAICompatProvider(LLMProvider):
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": float(temperature)}
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


@dataclass
class Rater:
    name: str
    provider: LLMProvider


# =========================
# 2) Utilities / IO
# =========================
def normalize_item(x: str) -> str:
    x = str(x).strip()
    x = x.replace("：", ":")
    m = re.match(r"^(R|C)\s*0*(\d+)\b", x, flags=re.I)
    if m:
        return m.group(1).upper() + str(int(m.group(2)))
    return x


def read_csv_smart(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def load_requirements_csv(path: str) -> pd.DataFrame:
    df = read_csv_smart(path)
    need = ["item", "text"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns {miss}; got columns={list(df.columns)}")
    df = df.copy()
    df["item"] = df["item"].apply(normalize_item)
    df["text"] = df["text"].astype(str)
    return df[["item", "text"]]


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

def _strip_code_fence(s: str) -> str:
    s = (s or "").strip()
    m = _JSON_FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    return s

def _extract_balanced_json(s: str) -> str:
    """
    Extract the first balanced JSON-like span starting from the first '{' or '['.
    If unclosed (model output truncated), return span to end and auto-close braces/brackets.
    This function ignores braces/brackets inside string literals.
    """
    i0 = None
    for i, ch in enumerate(s):
        if ch in "{[":
            i0 = i
            break
    if i0 is None:
        raise ValueError("No JSON start '{' or '[' found.")

    stack: List[str] = []
    in_str = False
    esc = False

    pairs = {"{": "}", "[": "]"}
    closers = set(pairs.values())

    for j in range(i0, len(s)):
        ch = s[j]

        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
            continue

        if ch in pairs:
            stack.append(ch)
            continue

        if ch in closers:
            if stack and pairs.get(stack[-1]) == ch:
                stack.pop()
                if not stack:
                    return s[i0 : j + 1]
            else:
                # mismatched closer, ignore
                continue

    # If we reached end and still unclosed, we assume truncation; auto-close
    span = s[i0:]
    if stack:
        span += "".join(pairs[o] for o in reversed(stack))
    return span

def _normalize_fullwidth_punct_outside_ascii_strings(s: str) -> str:
    """
    只在 ASCII 双引号字符串之外替换全角标点，避免破坏字符串内容。
    """
    mapping = {
        "：": ":",
        "，": ",",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    out = []
    in_str = False
    esc = False

    for ch in s:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
                out.append(ch)
            else:
                out.append(mapping.get(ch, ch))

    return "".join(out)


def _escape_unescaped_quotes_in_ascii_strings(s: str) -> str:
    """
    修复“字符串值内部出现未转义的 ASCII 双引号”的情况。
    规则：在字符串内遇到 `"` 时，如果它后面不是 , } ]（可跳过空白），则视为内容引号，转义成 \\"
    """
    out = []
    in_str = False
    esc = False
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
                esc = False
            i += 1
            continue

        # in string
        if esc:
            out.append(ch)
            esc = False
            i += 1
            continue

        if ch == "\\":
            out.append(ch)
            esc = True
            i += 1
            continue

        if ch == '"':
            # look ahead to decide closing vs inner quote
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j < n and s[j] in ",}]":
                # closing quote
                out.append(ch)
                in_str = False
            else:
                # inner quote -> escape it
                out.append('\\"')
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)



def extract_first_json(text: str) -> Dict[str, Any]:
    s = _strip_code_fence(text)

    # normalize common full-width quotes
    # DO NOT convert Chinese “ ” to ASCII quotes: it breaks valid JSON strings.
    s = s.replace("’", "'").replace("‘", "'")

    def _normalize_punct_outside_strings(x: str) -> str:
        # Convert full-width colon/comma outside strings to ASCII (helps Chinese outputs)
        out = []
        in_str = False
        esc = False
        for ch in x:
            if in_str:
                out.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    out.append(ch)
                    continue
                if ch == "：":
                    out.append(":")
                elif ch == "，":
                    out.append(",")
                elif ch == "｛":
                    out.append("{")
                elif ch == "｝":
                    out.append("}")
                elif ch == "［":
                    out.append("[")
                elif ch == "］":
                    out.append("]")
                else:
                    out.append(ch)
        return "".join(out)

    # Try multiple candidate starts to avoid picking '{' from explanation text
    starts = [m.start() for m in re.finditer(r"[\{\[]", s)]
    if not starts:
        raise ValueError("No JSON start found in model output.")

    last_err = None
    for idx in starts[:50]:  # cap to avoid pathological long texts
        try:
            span = _extract_balanced_json(s[idx:])
            span = _normalize_punct_outside_strings(span)

            # remove trailing commas before } or ]
            span = re.sub(r",\s*([}\]])", r"\1", span)

            # 1) strict JSON
            try:
                obj = json.loads(span)
            except json.JSONDecodeError:
                # 2) relaxed python-literal fallback
                s2 = span
                s2 = re.sub(r"\bnull\b", "None", s2, flags=re.I)
                s2 = re.sub(r"\btrue\b", "True", s2, flags=re.I)
                s2 = re.sub(r"\bfalse\b", "False", s2, flags=re.I)
                obj = ast.literal_eval(s2)

            if isinstance(obj, dict):
                return obj
            # If top-level is a list (some models output items array directly), wrap it.
            if isinstance(obj, list):
                return {"items": obj}
            raise ValueError(f"Top-level parsed value is not an object: {type(obj)}")

        except Exception as e:
            last_err = e
            continue

    raise ValueError(f"Failed to parse JSON from model output. Last error: {last_err}")





def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =========================
# 3) Prompting: scoring + discussion
# =========================
SCORING_SYSTEM = "你是严谨的需求质量评审专家（软件需求工程方向）。必须引用需求原文片段作为证据，改写必须可执行、原子、可测试。只输出JSON。"

SCORING_USER_TEMPLATE = """你将对下面这条需求进行质量评审（Likert 1-5，5为最好），并输出需要修改的问题清单（issues）。
taxonomy固定为四类：Unambiguous(U)、Understandable(A)、Correctness(C)、Verifiable(V)。

要求：
1) 每个维度给出 score(1-5) 与 confidence(0-1)；
    - 注意：必须输出“具体数值”，严禁写成 1-5 / 0-1 这种范围表达（那不是合法JSON）；
2) rationale 必须包含“证据片段”（引用原文中的关键短语/句子）并解释为何影响该维度；
3) suggestion 必须给出“可直接替换的改写文本”（更原子、可测试、无歧义），若你认为无需修改该维度，可给出 minimal suggestion（例如“保持不变”）；
4) issues：只列出你认为“需要改”的维度（taxonomy=U/A/C/V），每条包含 type、evidence、rewrite。
5) 只输出JSON，不要输出其他文本。
6)在任何字符串值中不得出现未转义的 ASCII 双引号 "；如需引用请使用中文引号“”或单引号''，或写成 \\\"。

需求条目：{item}
需求文本：{text}

输出JSON schema：
{{
  "item": "{item}",
  "scores": {{"U": 3, "A": 3, "C": 3, "V": 3}},
  "confidences": {{"U": 0.7, "A": 0.7, "C": 0.7, "V": 0.7}},
  "rationales": {{"U":"...","A":"...","C":"...","V":"..."}},
  "suggestions": {{"U":"...","A":"...","C":"...","V":"..."}},
  "issues": [
    {{"type":"U|A|C|V","evidence":"引用原文片段","rewrite":"改写文本"}}
  ]
}}
"""

DISCUSS_SYSTEM = "你是圆桌会议中的需求质量评审专家。你将看到其他模型对若干“需要更改的需求问题(issues)”的观点。请基于证据更新你的评分、置信度和改写建议。只输出JSON。"

DISCUSS_USER_TEMPLATE = """圆桌会议第 {round_idx} 轮。下面给出 Top-{topk} 个“需要更改的 issues”（按 influence=weight×recalibrated_conf 排序）。
注意：这里不强调维度分组，你需要对每个 issue 所属条目进行整体更新（但仍需输出U/A/C/V四维度的scores/confidences与issues）。

Top issues:
{issues_block}

你的任务：
1) 对每个条目，结合他人证据，更新你对该条目的 scores/confidences/rationales/suggestions；
2) 更新 issues 列表：只保留你认为“仍需要改”的维度，证据必须引用原文或他人给出的引用片段；
3) 如果你改变了观点，请说明是什么证据导致你改变，并相应调整confidence；
4) 只输出JSON，格式如下（items数组顺序任意，但必须覆盖给出的条目）：

{{
  "round": {round_idx},
  "items": [
    {{
      "item":"R1",
      "scores": {{"U": 3, "A": 3, "C": 3, "V": 3}},
      "confidences": {{"U": 0.7, "A": 0.7, "C": 0.7, "V": 0.7}},
      "rationales": {{"U":"...","A":"...","C":"...","V":"..."}},
      "suggestions": {{"U":"...","A":"...","C":"...","V":"..."}},
      "issues":[{{"type":"U|A|C|V","evidence":"...","rewrite":"..."}}]
    }}
  ]
}}
"""



def _append_jsonl(path: str, rec: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _repair_to_strict_json(rater: Rater, bad_text: str) -> str:
    fix_messages = [
        {"role": "system", "content": "你是严格JSON修复器。你只负责把输入文本转换为严格可解析的JSON。禁止输出除JSON之外的任何字符。"},
        {"role": "user", "content": "把下面文本修复为严格JSON（双引号、无尾逗号、无注释、无代码块）。只输出JSON：\n\n" + (bad_text or "")}
    ]
    return rater.provider.chat(fix_messages, temperature=0.0)

def _append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def call_llm_json(
    rater: Rater,
    messages: List[Dict[str, str]],
    temperature: float,
    out_dir: Optional[str] = None,
    stage: str = "unknown",
    round_idx: Optional[int] = None,
    item: Optional[str] = None,
) -> Dict[str, Any]:
    last_err = None
    log_path = os.path.join(out_dir, "raw_logs.jsonl") if out_dir else None

    for k in range(MAX_RETRIES + 1):
        txt = None
        fixed = None
        try:
            txt = rater.provider.chat(messages, temperature=temperature)

            # 1) 总是记录 raw 输出（这样“成功也有日志”）
            if log_path:
                _append_jsonl(log_path, {
                    "stage": stage,
                    "round": round_idx,
                    "item": item,
                    "rater": rater.name,
                    "attempt": k,
                    "temperature": float(temperature),
                    "kind": "raw",
                    "text": txt[:20000],
                })

            try:
                data = extract_first_json(txt)
                if log_path:
                    _append_jsonl(log_path, {
                        "stage": stage, "round": round_idx, "item": item,
                        "rater": rater.name, "attempt": k,
                        "kind": "parsed_ok"
                    })
                return data

            except Exception as e1:
                # 2) 解析失败 → 修复一次再解析
                fixed = _repair_to_strict_json(rater, txt)

                if log_path:
                    _append_jsonl(log_path, {
                        "stage": stage,
                        "round": round_idx,
                        "item": item,
                        "rater": rater.name,
                        "attempt": k,
                        "kind": "fixed",
                        "text": fixed[:20000],
                        "error_raw_parse": str(e1),
                    })

                data = extract_first_json(fixed)
                if log_path:
                    _append_jsonl(log_path, {
                        "stage": stage, "round": round_idx, "item": item,
                        "rater": rater.name, "attempt": k,
                        "kind": "fixed_parsed_ok"
                    })
                return data

        except Exception as e:
            last_err = e

            # 3) 失败也记一条
            if log_path:
                _append_jsonl(log_path, {
                    "stage": stage,
                    "round": round_idx,
                    "item": item,
                    "rater": rater.name,
                    "attempt": k,
                    "kind": "error",
                    "error": str(e),
                })

            # 4) 同时把完整文本落盘（便于手工排查）
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                ts = int(time.time() * 1000)

                if txt:
                    p1 = os.path.join(out_dir, f"bad_json_{stage}_{rater.name}_{item or 'NA'}_raw_a{k}_{ts}.txt")
                    with open(p1, "w", encoding="utf-8") as f:
                        f.write(txt)

                if fixed:
                    p2 = os.path.join(out_dir, f"bad_json_{stage}_{rater.name}_{item or 'NA'}_fixed_a{k}_{ts}.txt")
                    with open(p2, "w", encoding="utf-8") as f:
                        f.write(fixed)

            time.sleep(RETRY_SLEEP)

    raise RuntimeError(f"[{rater.name}] failed after retries: {last_err}")





def score_requirement(rater: Rater, item: str, text: str, out_dir: Optional[str] = None) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SCORING_SYSTEM},
        {"role": "user", "content": SCORING_USER_TEMPLATE.format(item=item, text=text)},
    ]
    # IMPORTANT: pass out_dir so raw/fixed/bad_json logs are persisted
    data = call_llm_json(rater, messages, temperature=TEMPERATURE_SCORE, out_dir=out_dir, stage="score", item=item)

    # --- [MIN PATCH] unwrap payload / alias keys / dim keys ---
    # 0) unwrap common wrapper keys
    if isinstance(data, dict):
        for k in ("result", "data", "output", "payload"):
            if isinstance(data.get(k), dict):
                data = data[k]
                break

    # 1) discuss-style {"items":[...]} OR top-level list -> pick the matching item
    items_list = None
    if isinstance(data, list):
        items_list = data
    elif isinstance(data, dict) and isinstance(data.get("items"), list):
        items_list = data["items"]

    if isinstance(items_list, list):
        want = normalize_item(item)
        target = None
        for it in items_list:
            if isinstance(it, dict) and normalize_item(it.get("item", "")) == want:
                target = it
                break
        if target is None and len(items_list) == 1 and isinstance(items_list[0], dict):
            target = items_list[0]
        if target is not None:
            data = target

    # 2) Alias compatibility: score/confidence -> scores/confidences
    if isinstance(data, dict):
        if "scores" not in data and "score" in data:
            data["scores"] = data["score"]
        if "confidences" not in data and "confidence" in data:
            data["confidences"] = data["confidence"]

        # 3) (optional but recommended) remap dimension full names -> U/A/C/V
        DIM_MAP = {
            "Understandable": "U",
            "Unambiguous": "A",
            "Correctness": "C",
            "Verifiable": "V",
            "可理解性": "U",
            "无歧义": "A",
            "正确性": "C",
            "可验证性": "V",
        }

        def _remap_dims(d):
            if not isinstance(d, dict):
                return d
            return {DIM_MAP.get(k, k): v for k, v in d.items()}

        if "scores" in data:
            data["scores"] = _remap_dims(data["scores"])
        if "confidences" in data:
            data["confidences"] = _remap_dims(data["confidences"])
    # --- [MIN PATCH END] ---

    # minimal validation
    if "scores" not in data or "confidences" not in data:
        raise ValueError(f"[{rater.name}] invalid output: missing scores/confidences")
    for sdim in ["U", "A", "C", "V"]:
        if sdim not in data["scores"] or sdim not in data["confidences"]:
            raise ValueError(f"[{rater.name}] invalid output: missing {sdim}")
    return data


# =========================
# 4) Build state tables
# =========================
def flatten_scoring_output(rater: str, item: str, data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - rating_long: rows per dim (score/confidence/rationale/suggestion)
      - issues_long: rows per issue (type/evidence/rewrite)
    """
    rows = []
    for sdim in ["U", "A", "C", "V"]:
        rows.append({
            "rater": rater,
            "item": item,
            "dim": sdim,
            "dimension": SHORT_DIM[sdim],
            "score": float(data["scores"][sdim]),
            "confidence": float(data["confidences"][sdim]),
            "rationale": str(data.get("rationales", {}).get(sdim, ""))[:4000],
            "suggestion": str(data.get("suggestions", {}).get(sdim, ""))[:4000],
        })
    rating_long = pd.DataFrame(rows)

    issues = data.get("issues", []) or []
    irows = []
    for it in issues:
        t = str(it.get("type", "")).strip().upper()
        if t not in ["U", "A", "C", "V"]:
            continue
        irows.append({
            "rater": rater,
            "item": item,
            "type": t,
            "dimension": SHORT_DIM[t],
            "evidence": str(it.get("evidence", ""))[:2000],
            "rewrite": str(it.get("rewrite", ""))[:4000],
        })
    issues_long = pd.DataFrame(irows) if irows else pd.DataFrame(columns=["rater","item","type","dimension","evidence","rewrite"])
    return rating_long, issues_long


def weighted_team_score(ratings_long: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Team score per item×dim using w * recalibrated_conf.
    """
    df = ratings_long.copy()
    df["w"] = df["rater"].map(weights).fillna(0.0)
    df["pc"] = df["confidence"].apply(recalibrate_conf)
    df["cw"] = df["w"] * df["pc"]

    def _agg(g: pd.DataFrame) -> pd.Series:
        denom = float(g["cw"].sum())
        if denom <= 0:
            val = float(g["score"].mean())
        else:
            val = float((g["cw"] * g["score"]).sum() / denom)
        return pd.Series({"team_score": val})

    out = df.groupby(["item","dim"], as_index=False).apply(_agg).reset_index(drop=True)
    return out


def compute_consensus_stats(ratings_long: pd.DataFrame) -> pd.DataFrame:
    """
    For hot-cell detection: range/IQR per item×dim across raters.
    """
    df = ratings_long.copy()

    def _agg(g: pd.DataFrame) -> pd.Series:
        scores = g["score"].to_numpy(dtype=float)
        med = float(np.median(scores))
        rng = float(np.max(scores) - np.min(scores))
        iqr = float(np.subtract(*np.percentile(scores, [75, 25])))
        return pd.Series({"median": med, "range": rng, "IQR": iqr})

    c = df.groupby(["item","dim"], as_index=False).apply(_agg).reset_index(drop=True)
    return c


def mark_hot_cells(ratings_long: pd.DataFrame) -> pd.DataFrame:
    """
    Hot cell = (range>=2) or (IQR>=1) or (max abs delta to median >=2)
    """
    cons = compute_consensus_stats(ratings_long)
    df = ratings_long.merge(cons, on=["item","dim"], how="left")
    df["abs_delta"] = (df["score"] - df["median"]).abs()

    max_abs = df.groupby(["item","dim"], as_index=False)["abs_delta"].max().rename(columns={"abs_delta":"max_abs_delta"})
    cons2 = cons.merge(max_abs, on=["item","dim"], how="left").fillna({"max_abs_delta": 0.0})

    cons2["is_hot"] = (
        (cons2["range"] >= HOTCELL_RULES["range_ge"]) |
        (cons2["IQR"] >= HOTCELL_RULES["iqr_ge"]) |
        (cons2["max_abs_delta"] >= HOTCELL_RULES["abs_delta_ge"])
    )
    return cons2


def issue_support_scores(issues_long: pd.DataFrame, ratings_long: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    For each issue candidate (item,type): S = sum_{raters who flagged} w * recalibrated_conf(type)
    """
    # map each rater's confidence for that type
    conf_map = ratings_long[["rater","item","dim","confidence"]].copy()
    conf_map.rename(columns={"dim":"type"}, inplace=True)  # type in U/A/C/V
    conf_map["pc"] = conf_map["confidence"].apply(recalibrate_conf)
    conf_map["w"] = conf_map["rater"].map(weights).fillna(0.0)
    conf_map["influence"] = conf_map["w"] * conf_map["pc"]

    # only count raters who flagged that issue
    flagged = issues_long[["rater","item","type"]].drop_duplicates()
    joined = flagged.merge(conf_map[["rater","item","type","influence"]], on=["rater","item","type"], how="left").fillna({"influence":0.0})

    s = joined.groupby(["item","type"], as_index=False)["influence"].sum().rename(columns={"influence":"S_issue"})
    s["dimension"] = s["type"].map(SHORT_DIM)
    return s.sort_values("S_issue", ascending=False).reset_index(drop=True)


def build_topk_issues_block(
    topk: pd.DataFrame,
    req_df: pd.DataFrame,
    issues_long: pd.DataFrame,
    ratings_long: pd.DataFrame,
    weights: Dict[str, float],
    k: int
) -> str:
    """
    Build human-readable block for LLM prompt, sorted by S_issue desc, with per-issue supporting views ordered by influence.
    """
    req_map = dict(zip(req_df["item"], req_df["text"]))

    # compute per-rater influence for each issue type
    conf_map = ratings_long[["rater","item","dim","confidence"]].copy()
    conf_map.rename(columns={"dim":"type"}, inplace=True)
    conf_map["pc"] = conf_map["confidence"].apply(recalibrate_conf)
    conf_map["w"] = conf_map["rater"].map(weights).fillna(0.0)
    conf_map["influence"] = conf_map["w"] * conf_map["pc"]

    blocks = []
    used_items = set()
    for _, row in topk.head(k).iterrows():
        item = row["item"]
        typ = row["type"]
        S = float(row["S_issue"])
        used_items.add(item)

        text = req_map.get(item, "")
        # collect raters who flagged this issue
        sub_iss = issues_long[(issues_long["item"] == item) & (issues_long["type"] == typ)].copy()
        if sub_iss.empty:
            continue
        sub_iss = sub_iss.merge(conf_map[["rater","item","type","influence"]], on=["rater","item","type"], how="left").fillna({"influence":0.0})
        sub_iss = sub_iss.sort_values("influence", ascending=False)

        # attach score info for that dim

        dim_scores = ratings_long[(ratings_long["item"] == item) & (ratings_long["dim"] == typ)][
            ["rater", "score", "confidence", "rationale", "suggestion"]
        ].copy()
        # avoid cartesian product: join on (rater,item,type)
        dim_scores["item"] = item
        dim_scores["type"] = typ
        dim_scores = dim_scores.merge(
            conf_map[["rater", "item", "type", "influence"]],
            on = ["rater", "item", "type"],
            how = "left"
        )

        dim_scores = dim_scores.sort_values("influence", ascending=False)

        views = []
        for _, v in dim_scores.iterrows():
            views.append(
                f"- {v['rater']} | score={v['score']}, conf={v['confidence']:.2f}, infl≈{float(v['influence']):.3f}\n"
                f"  rationale: {str(v['rationale'])[:300]}\n"
                f"  suggestion: {str(v['suggestion'])[:200]}"
            )

        ev = []
        for _, it in sub_iss.iterrows():
            ev.append(
                f"- {it['rater']} infl≈{float(it['influence']):.3f}\n"
                f"  evidence: {str(it['evidence'])[:200]}\n"
                f"  rewrite: {str(it['rewrite'])[:200]}"
            )

        blocks.append(
            f"[Issue] item={item} type={typ} (dim={SHORT_DIM[typ]}) S_issue={S:.3f}\n"
            f"Requirement: {text[:600]}\n"
            f"Rater views (ordered by influence):\n" + "\n".join(views[:6]) + "\n"
            f"Flagged issue details:\n" + "\n".join(ev[:6]) + "\n"
        )

    return "\n\n".join(blocks)


# =========================
# 5) Fixed weights: load or compute from analyze report + labels
# =========================
def load_weights_csv(path: str) -> Dict[str, float]:
    df = read_csv_smart(path)
    need = ["rater","weight"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns {miss}; got {list(df.columns)}")
    df = df.copy()
    df["rater"] = df["rater"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"])
    s = float(df["weight"].sum())
    if s <= 0:
        raise ValueError("weights sum <=0")
    df["weight"] = df["weight"] / s
    return dict(zip(df["rater"], df["weight"]))


def load_model_labels(path: str) -> pd.DataFrame:
    df = read_csv_smart(path)
    need = ["rater","bad_global","good_global"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns {miss}")
    df = df.copy()
    df["rater"] = df["rater"].astype(str).str.strip()
    df["bad_global"] = pd.to_numeric(df["bad_global"], errors="coerce").fillna(0).astype(int)
    df["good_global"] = pd.to_numeric(df["good_global"], errors="coerce").fillna(0).astype(int)
    return df[["rater","bad_global","good_global"]]


def load_hotcell_labels(path: str) -> pd.DataFrame:
    df = read_csv_smart(path)
    need = ["rater","item","dimension","label"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns {miss}")
    df = df.copy()
    df["rater"] = df["rater"].astype(str).str.strip()
    df["item"] = df["item"].apply(normalize_item)
    df["dimension"] = df["dimension"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    # normalize dimension to U/A/C/V if needed
    # accept "Unambiguous" etc
    dim2 = []
    for x in df["dimension"].tolist():
        if x.upper() in ["U","A","C","V"]:
            dim2.append(x.upper())
        elif x in DIMENSIONS:
            dim2.append(DIM_SHORT[x])
        else:
            # fallback: first letter
            dim2.append(x[:1].upper())
    df["dim"] = dim2
    df["is_bad"] = df["label"].isin(["bad","drop"]).astype(int)
    df["is_good"] = df["label"].isin(["good","keep"]).astype(int)
    return df[["rater","item","dim","is_bad","is_good"]]


def minmax_norm(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mn = float(np.nanmin(x.to_numpy())) if np.isfinite(np.nanmin(x.to_numpy())) else np.nan
    mx = float(np.nanmax(x.to_numpy())) if np.isfinite(np.nanmax(x.to_numpy())) else np.nan
    if pd.isna(mn) or pd.isna(mx):
        return pd.Series(np.nan, index=s.index)
    if mx == mn:
        return pd.Series(0.0, index=s.index)
    return (x - mn) / (mx - mn)


def compute_weights_from_analyze_report(
    report_xlsx: str,
    raters: List[str],
    model_labels_csv: Optional[str],
    hotcell_labels_csv: Optional[str],
) -> Dict[str, float]:
    """
    Uses sheets from analyze_llm_likert_scores output (req-level):
      - avgdisg_req: avg_distance_to_others
      - loo_req: contribution(G-G_minus)
      - rater_summary_req: bias_to_median (=> bias_abs)
      - dev_summary_req: mean_abs_delta
    """
    xl = pd.ExcelFile(report_xlsx)

    # --- PATCH START: accept both avgdist_req (new) and avgdisg_req (old) ---
    def _pick_sheet(cands):
        for s in cands:
            if s in xl.sheet_names:
                return s
        raise ValueError(f"{report_xlsx} missing sheets {cands}. Found: {xl.sheet_names}")

    avgdist_sh = _pick_sheet(["avgdist_req", "avgdisg_req"])
    # --- PATCH END ---

    required_sheets = ["loo_req", "rater_summary_req", "dev_summary_req"]
    for sh in required_sheets:
        if sh not in xl.sheet_names:
            raise ValueError(f"{report_xlsx} missing sheet '{sh}'. Found: {xl.sheet_names}")

    avgdisg = xl.parse(avgdist_sh)
    loo = xl.parse("loo_req")
    rs = xl.parse("rater_summary_req")
    devs = xl.parse("dev_summary_req")

    # Build table
    df = pd.DataFrame({"rater": raters})
    # avg_distance_to_others
    if "avg_distance_to_others" not in avgdisg.columns:
        raise ValueError("avgdisg_req missing avg_distance_to_others")
    df = df.merge(avgdisg[["rater","avg_distance_to_others"]], on="rater", how="left")

    # loo contribution
    if "contribution(G-G_minus)" not in loo.columns:
        raise ValueError("loo_req missing contribution(G-G_minus)")
    df = df.merge(loo[["rater","contribution(G-G_minus)"]], on="rater", how="left")

    # bias abs
    if "bias_to_median" not in rs.columns:
        raise ValueError("rater_summary_req missing bias_to_median")
    rs2 = rs[["rater","bias_to_median"]].copy()
    rs2["bias_abs"] = rs2["bias_to_median"].abs()
    df = df.merge(rs2[["rater","bias_abs"]], on="rater", how="left")

    # mean abs delta
    if "mean_abs_delta" not in devs.columns:
        raise ValueError("dev_summary_req missing mean_abs_delta")
    df = df.merge(devs[["rater","mean_abs_delta"]], on="rater", how="left")

    # normalize metrics and compute OutlierIndex
    oi = pd.Series(0.0, index=df.index, dtype=float)
    for m, a in OUTLIER_METRICS_WEIGHTS.items():
        df[f"norm_{m}"] = minmax_norm(df[m])
        oi += a * df[f"norm_{m}"].fillna(0.0)
    df["OI"] = oi

    # human labels
    bad_global = pd.Series(0.0, index=df.index)
    if model_labels_csv and os.path.exists(model_labels_csv):
        ml = load_model_labels(model_labels_csv)
        df = df.merge(ml[["rater","bad_global"]], on="rater", how="left")
        bad_global = df["bad_global"].fillna(0.0).astype(float)
    else:
        df["bad_global"] = 0

    bad_hot_rate = pd.Series(0.0, index=df.index)
    good_hot_rate = pd.Series(0.0, index=df.index)
    if hotcell_labels_csv and os.path.exists(hotcell_labels_csv):
        hl = load_hotcell_labels(hotcell_labels_csv)
        agg = hl.groupby("rater", as_index=False).agg(
            bad_hot=("is_bad","sum"),
            good_hot=("is_good","sum"),
            total=("is_bad","count")
        )
        agg["bad_hot_rate"] = agg["bad_hot"] / agg["total"].replace(0, np.nan)
        agg["good_hot_rate"] = agg["good_hot"] / agg["total"].replace(0, np.nan)
        df = df.merge(agg[["rater","bad_hot_rate","good_hot_rate"]], on="rater", how="left")
        bad_hot_rate = df["bad_hot_rate"].fillna(0.0).astype(float)
        good_hot_rate = df["good_hot_rate"].fillna(0.0).astype(float)
    else:
        df["bad_hot_rate"] = 0.0
        df["good_hot_rate"] = 0.0

    # Fixed weight formula (monotonic, interpretable)
    raw = np.exp(-df["OI"].fillna(0.0).to_numpy(dtype=float))
    raw *= np.exp(-LAMBDA_BAD_GLOBAL * bad_global.to_numpy(dtype=float))
    raw *= np.exp(-LAMBDA_BAD_HOT * bad_hot_rate.to_numpy(dtype=float))
    raw *= (1.0 + LAMBDA_GOOD_HOT * good_hot_rate.to_numpy(dtype=float))

    raw = np.maximum(raw, 1e-9)
    w = raw / raw.sum()

    df["weight"] = w
    # normalize again guard
    df["weight"] = df["weight"] / df["weight"].sum()

    return dict(zip(df["rater"], df["weight"]))


# =========================
# 6) Roundtable loop
# =========================
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0


def team_issues_set(issue_scores: pd.DataFrame, theta: float) -> set:
    s = issue_scores[issue_scores["S_issue"] >= theta][["item","type"]]
    return set(map(tuple, s.to_records(index=False)))


def run_roundtable(
    raters: List[Rater],
    req_df: pd.DataFrame,
    weights: Dict[str, float],
    max_rounds: int,
    topk_issues: int,
    theta: float,
    eps_score: float,
    tau_jacc: float,
    out_dir: str
) -> Dict[str, Any]:
    """
    Returns dict with final tables + history
    """
    req_map = dict(zip(req_df["item"], req_df["text"]))
    ensure_dir(out_dir)

    # ---------------- initial scoring
    all_ratings = []
    all_issues = []

    raw_logs = []

    for _, rrow in req_df.iterrows():
        item = rrow["item"]
        text = rrow["text"]
        for r in raters:
            data = score_requirement(r, item, text, out_dir=out_dir)
            raw_logs.append({"stage":"score", "rater":r.name, "item":item, "raw":data})
            rating_long, issues_long = flatten_scoring_output(r.name, item, data)
            all_ratings.append(rating_long)
            all_issues.append(issues_long)

    ratings_long = pd.concat(all_ratings, ignore_index=True)
    issues_long = pd.concat(all_issues, ignore_index=True)

    # store round 0
    ratings_by_round = {0: ratings_long.copy()}
    issues_by_round = {0: issues_long.copy()}
    issue_scores_by_round = {}
    team_scores_by_round = {}

    # round history
    hist = []

    prev_team_scores = None
    prev_issue_set = None

    # ---------------- iterations
    for t in range(0, max_rounds + 1):
        cur_ratings = ratings_by_round[t]
        cur_issues = issues_by_round[t]

        # compute issue scores and team scores
        issue_scores = issue_support_scores(cur_issues, cur_ratings, weights)
        team_scores = weighted_team_score(cur_ratings, weights)

        issue_scores_by_round[t] = issue_scores
        team_scores_by_round[t] = team_scores

        cur_issue_set = team_issues_set(issue_scores, theta)

        # convergence check (skip at t=0)
        score_converged = False
        issue_converged = False
        max_score_diff = None
        jacc = None

        if prev_team_scores is not None:
            merged = team_scores.merge(prev_team_scores, on=["item","dim"], how="inner", suffixes=("_cur","_prev"))
            diffs = (merged["team_score_cur"] - merged["team_score_prev"]).abs()
            max_score_diff = float(diffs.max()) if not diffs.empty else 0.0
            score_converged = (max_score_diff <= eps_score)

        if prev_issue_set is not None:
            jacc = jaccard(cur_issue_set, prev_issue_set)
            issue_converged = (jacc >= tau_jacc)

        hist.append({
            "round": t,
            "n_issue_candidates": int(len(issue_scores)),
            "n_team_issues_ge_theta": int(sum(issue_scores["S_issue"] >= theta)) if len(issue_scores) else 0,
            "score_converged": bool(score_converged),
            "issue_converged": bool(issue_converged),
            "max_score_diff": max_score_diff,
            "issue_jaccard": jacc,
        })

        # stop if any (after at least 1 iteration computed)
        if t > 0 and (score_converged or issue_converged):
            break
        if t == max_rounds:
            break

        # choose topK issue candidates to discuss (any flagged issue, ranked by S_issue)
        topk = issue_scores.head(topk_issues).copy()
        if topk.empty:
            break

        issues_block = build_topk_issues_block(topk, req_df, cur_issues, cur_ratings, weights, k=topk_issues)

        # in discussion: each rater updates only the involved items
        discuss_items = sorted(set(topk["item"].tolist()))
        next_ratings_rows = []
        next_issues_rows = []

        for r in raters:
            msg = [
                {"role":"system", "content": DISCUSS_SYSTEM},
                {"role":"user", "content": DISCUSS_USER_TEMPLATE.format(
                    round_idx=t+1,
                    topk=topk_issues,
                    issues_block=issues_block
                )}
            ]
            upd = call_llm_json(
                r, msg,
                temperature=TEMPERATURE_DISCUSS,
                out_dir=out_dir,
                stage="discuss",
                round_idx=t + 1,
                item=",".join(discuss_items[:5])  # 只是方便定位，可随便
            )

            raw_logs.append({"stage":"discuss", "round":t+1, "rater":r.name, "raw":upd})

            items_out = upd.get("items", []) or []
            # Build a map for updated items
            upd_map = {}
            for it in items_out:
                ii = normalize_item(it.get("item",""))
                if ii:
                    upd_map[ii] = it

            # For items not provided by rater, keep old state (for discussed items)
            for item in discuss_items:
                if item in upd_map:
                    data = {
                        "scores": upd_map[item].get("scores", {}),
                        "confidences": upd_map[item].get("confidences", {}),
                        "rationales": upd_map[item].get("rationales", {}),
                        "suggestions": upd_map[item].get("suggestions", {}),
                        "issues": upd_map[item].get("issues", []),
                    }
                    # validation + fix keys maybe already U/A/C/V
                    # If model outputs full names, try map
                    if set(data["scores"].keys()) & set(DIM_SHORT.keys()):
                        # convert to U/A/C/V if needed
                        new_scores = {}
                        for k0, v0 in data["scores"].items():
                            if k0 in DIM_SHORT:
                                new_scores[DIM_SHORT[k0]] = v0
                        data["scores"] = new_scores
                    if set(data["confidences"].keys()) & set(DIM_SHORT.keys()):
                        new_conf = {}
                        for k0, v0 in data["confidences"].items():
                            if k0 in DIM_SHORT:
                                new_conf[DIM_SHORT[k0]] = v0
                        data["confidences"] = new_conf

                    # Attach item for flatten
                    data2 = {"scores": data["scores"], "confidences": data["confidences"],
                             "rationales": data.get("rationales", {}), "suggestions": data.get("suggestions", {}),
                             "issues": data.get("issues", [])}
                    rl, il = flatten_scoring_output(r.name, item, data2)
                    next_ratings_rows.append(rl)
                    next_issues_rows.append(il)
                else:
                    # keep previous rows for this rater+item
                    keep_r = cur_ratings[(cur_ratings["rater"]==r.name) & (cur_ratings["item"]==item)]
                    keep_i = cur_issues[(cur_issues["rater"]==r.name) & (cur_issues["item"]==item)]
                    next_ratings_rows.append(keep_r.copy())
                    next_issues_rows.append(keep_i.copy())

            # For non-discussed items, keep old state
            other_r = cur_ratings[(cur_ratings["rater"]==r.name) & (~cur_ratings["item"].isin(discuss_items))]
            other_i = cur_issues[(cur_issues["rater"]==r.name) & (~cur_issues["item"].isin(discuss_items))]
            next_ratings_rows.append(other_r.copy())
            next_issues_rows.append(other_i.copy())

        ratings_by_round[t+1] = pd.concat(next_ratings_rows, ignore_index=True)
        issues_by_round[t+1] = pd.concat(next_issues_rows, ignore_index=True)

        prev_team_scores = team_scores
        prev_issue_set = cur_issue_set

    # choose final round
    final_round = max(ratings_by_round.keys())
    final_ratings = ratings_by_round[final_round]
    final_issues = issues_by_round[final_round]
    final_issue_scores = issue_scores_by_round[final_round]
    final_team_scores = team_scores_by_round[final_round]

    # produce final problems list based on issues score >= theta
    probs = final_issue_scores[final_issue_scores["S_issue"] >= theta].copy()
    probs = probs.merge(req_df, on="item", how="left")

    # pick best evidence/rewrite from highest influence supporter
    # compute supporter influences per (item,type,rater)
    conf_map = final_ratings[["rater","item","dim","confidence"]].copy()
    conf_map.rename(columns={"dim":"type"}, inplace=True)
    conf_map["pc"] = conf_map["confidence"].apply(recalibrate_conf)
    conf_map["w"] = conf_map["rater"].map(weights).fillna(0.0)
    conf_map["influence"] = conf_map["w"] * conf_map["pc"]

    supports = final_issues.merge(conf_map[["rater","item","type","influence"]], on=["rater","item","type"], how="left").fillna({"influence":0.0})
    supports = supports.sort_values("influence", ascending=False)

    best_rows = []
    for (item, typ), g in supports.groupby(["item","type"]):
        g = g.sort_values("influence", ascending=False)
        top = g.iloc[0]
        # also list top supporters
        top_supporters = g.head(5)[["rater","influence"]].to_dict("records")
        best_rows.append({
            "item": item,
            "type": typ,
            "dimension": SHORT_DIM[typ],
            "best_evidence": top["evidence"],
            "best_rewrite": top["rewrite"],
            "top_supporters": json.dumps(top_supporters, ensure_ascii=False),
        })
    best = pd.DataFrame(best_rows)

    probs = probs.merge(best, on=["item","type","dimension"], how="left")

    # save raw logs
    with open(os.path.join(out_dir, "raw_logs.jsonl"), "w", encoding="utf-8") as f:
        for rec in raw_logs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "final_round": final_round,
        "ratings_by_round": ratings_by_round,
        "issues_by_round": issues_by_round,
        "issue_scores_by_round": issue_scores_by_round,
        "team_scores_by_round": team_scores_by_round,
        "history": pd.DataFrame(hist),
        "final_problems": probs.sort_values("S_issue", ascending=False).reset_index(drop=True),
    }


# =========================
# 7) Main / CLI
# =========================
def load_raters_from_models_json(models_json: str) -> List[Rater]:
    with open(models_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    raters_cfg = cfg.get("raters", [])
    if not raters_cfg:
        raise ValueError("models.json missing raters")
    raters = []
    for rc in raters_cfg:
        name = str(rc["name"]).strip()
        base_url = str(rc["base_url"]).strip()
        model = str(rc["model"]).strip()
        api_key_env = str(rc.get("api_key_env","")).strip()
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        if not api_key:
            raise ValueError(f"Missing API key for rater '{name}'. Set env '{api_key_env}'.")
        prov = OpenAICompatProvider(base_url=base_url, api_key=api_key, model=model)
        raters.append(Rater(name=name, provider=prov))
    return raters


def export_excel(out_xlsx: str, req_df: pd.DataFrame, weights: Dict[str,float], result: Dict[str,Any]):
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        req_df.to_excel(writer, sheet_name="requirements", index=False)

        wdf = pd.DataFrame({"rater": list(weights.keys()), "weight": list(weights.values())}).sort_values("weight", ascending=False)
        wdf.to_excel(writer, sheet_name="weights", index=False)

        hist = result["history"]
        hist.to_excel(writer, sheet_name="round_history", index=False)

        # rounds
        for t, df in result["ratings_by_round"].items():
            df.to_excel(writer, sheet_name=f"ratings_r{t}", index=False)
        for t, df in result["issues_by_round"].items():
            df.to_excel(writer, sheet_name=f"issues_r{t}", index=False)
        for t, df in result["issue_scores_by_round"].items():
            df.to_excel(writer, sheet_name=f"issue_scores_r{t}", index=False)
        for t, df in result["team_scores_by_round"].items():
            df.to_excel(writer, sheet_name=f"team_scores_r{t}", index=False)

        # final problems
        result["final_problems"].to_excel(writer, sheet_name="final_problems", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", required=True, help="requirements.csv with columns item,text")
    ap.add_argument("--models", required=True, help="models.json (OpenAI-compatible endpoints)")
    ap.add_argument("--out", default="roundtable_report.xlsx", help="output Excel file")
    ap.add_argument("--out_dir", default="roundtable_logs", help="output directory for logs")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK_ISSUES, help="top-K issues to discuss each round")
    ap.add_argument("--rounds", type=int, default=DEFAULT_MAX_ROUNDS, help="max rounds")
    ap.add_argument("--theta_ratio", type=float, default=DEFAULT_THETA_RATIO, help="Theta ratio * sum(weights)")

    # convergence
    ap.add_argument("--eps_score", type=float, default=DEFAULT_EPS_SCORE, help="score convergence epsilon")
    ap.add_argument("--tau_jacc", type=float, default=DEFAULT_TAU_JACCARD, help="issue-set convergence Jaccard threshold")

    # weights sources
    ap.add_argument("--weights_csv", default="", help="optional weights.csv (rater,weight)")
    ap.add_argument("--analyze_xlsx", default="", help="optional llm_rating_report.xlsx from analyze_llm_likert_scores")
    ap.add_argument("--model_labels", default="", help="optional model_labels.csv")
    ap.add_argument("--hotcell_labels", default="", help="optional hotcell_labels.csv")

    args = ap.parse_args()

    req_df = load_requirements_csv(args.requirements)
    raters = load_raters_from_models_json(args.models)
    rater_names = [r.name for r in raters]

    # fixed weights
    if args.weights_csv:
        weights = load_weights_csv(args.weights_csv)
    else:
        if not args.analyze_xlsx:
            raise ValueError("Provide either --weights_csv or --analyze_xlsx (with optional labels).")
        weights = compute_weights_from_analyze_report(
            report_xlsx=args.analyze_xlsx,
            raters=rater_names,
            model_labels_csv=args.model_labels if args.model_labels else None,
            hotcell_labels_csv=args.hotcell_labels if args.hotcell_labels else None,
        )

    # normalize weights over present raters
    wsum = sum(weights.get(r, 0.0) for r in rater_names)
    if wsum <= 0:
        raise ValueError("weights sum <=0 over configured raters")
    weights = {r: weights.get(r, 0.0) / wsum for r in rater_names}

    theta = float(args.theta_ratio) * sum(weights.values())  # if normalized -> theta_ratio

    result = run_roundtable(
        raters=raters,
        req_df=req_df,
        weights=weights,
        max_rounds=int(args.rounds),
        topk_issues=int(args.topk),
        theta=theta,
        eps_score=float(args.eps_score),
        tau_jacc=float(args.tau_jacc),
        out_dir=args.out_dir
    )

    export_excel(args.out, req_df, weights, result)
    print(f"Saved Excel: {args.out}")
    print(f"Saved logs: {args.out_dir}/raw_logs.jsonl")
    print("Final problems (top 10):")
    print(result["final_problems"][["item","type","dimension","S_issue"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     for p in [
#         "bad_json_score_deepseek_R1_raw_a1_1768654221768.txt",
#         "bad_json_score_deepseek_R1_raw_a2_1768654272248.txt",
#     ]:
#         with open(p, "r", encoding="utf-8") as f:
#             t = f.read()
#         obj = extract_first_json(t)
#         print(p, "OK", obj.keys())
