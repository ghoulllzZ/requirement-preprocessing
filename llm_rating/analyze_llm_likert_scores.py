# -*- coding: utf-8 -*-
"""
Extended unified-format analyzer for multi-LLM Likert scores.

Input per rater:
1) Per-requirement CSV (wide):
   item,Unambiguous,Understandable,Correctness,Verifiable
2) Document-level CSV (long):
   dimension,score
   dimension ∈ {Internal Consistency, Non-redundancy, Completeness, Conciseness}

Outputs:
- llm_rating_report.xlsx (extended sheets)

Install:
  pip install pandas numpy scipy openpyxl
"""

import os
import re
import itertools
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, kendalltau


# ----------------------------
# CONFIG: edit file names here
# ----------------------------
RATERS = {
    "gpt": {
        "req": "gpt_per-requirement_rating.csv",
        "doc": "gpt_document-wide_rating.csv",
    },
    "deepseek": {
        "req": "deepseek_per-requirement_rating.csv",
        "doc": "deepseek_document-wide_rating.csv",
    },
    "doubao": {
        "req": "doubao_per-requirement_rating.csv",
        "doc": "doubao_document-wide_rating.csv",
    },
    "qwen": {
        "req": "qwen_per-requirement_rating.csv",
        "doc": "qwen_document-wide_rating.csv",
    },
}

REQ_DIMS = ["Unambiguous", "Understandable", "Correctness", "Verifiable"]
DOC_DIMS = ["Internal Consistency", "Non-redundancy", "Completeness", "Conciseness"]

CORE_ITEMS = [f"R{i}" for i in range(1, 9)]  # optional: R1..R8
EXT_ITEM = "C1"


# ----------------------------
# Utilities
# ----------------------------
def read_csv_auto(path: str) -> pd.DataFrame:
    """Try common encodings (UTF-8 w/wo BOM, then GB encodings)."""
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "read_csv_auto", b"", 0, 1,
        f"Cannot decode {path}. Please re-save as 'CSV UTF-8' in Excel."
    )


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def normalize_item_label(s: str):
    """Map various item labels to standard IDs: R1..R8, C1; otherwise keep original."""
    s = str(s).strip()
    if s in ["", "...", "…", "None", "nan", "NaN"]:
        return None

    # C1 / extensibility
    if re.search(r"\bC1\b", s, re.I) or ("扩展" in s) or ("可扩展" in s):
        return "C1"

    # Pure number prefix -> R#
    m = re.match(r"^\s*(\d+)", s)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 8:
            return f"R{num}"
        if num == 9:
            return "C1"

    # R# prefix
    m = re.match(r"^\s*R(\d+)", s, re.I)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 8:
            return f"R{num}"
        if num == 9:
            return "C1"

    return s


def require_columns(df: pd.DataFrame, cols, context: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{context}] Missing columns {missing}. Actual columns: {df.columns.tolist()}")


def to_numeric_1to5(series: pd.Series, context: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # keep only 1..5
    bad = s.notna() & ((s < 1) | (s > 5))
    if bad.any():
        raise ValueError(f"[{context}] Found scores out of 1..5 range. Examples:\n{series[bad].head(10)}")
    return s


# ----------------------------
# Load unified-format inputs
# ----------------------------
def load_req_unified(path: str) -> pd.DataFrame:
    """
    Expect wide format:
      item,Unambiguous,Understandable,Correctness,Verifiable
    Returns wide table with columns: std_item + REQ_DIMS
    If rater splits items (e.g., R6a/R6b), aggregate by MIN.
    """
    df = read_csv_auto(path)
    df = clean_columns(df)

    # Allow aliases for item column
    if "item" not in df.columns:
        for alias in ["需求条目", "需求编号", "功能点（对应原需求）", "功能点(对应原需求)"]:
            if alias in df.columns:
                df = df.rename(columns={alias: "item"})
                break

    require_columns(df, ["item"] + REQ_DIMS, context=f"REQ {os.path.basename(path)}")

    df["std_item"] = df["item"].apply(normalize_item_label)

    # numeric coercion
    for d in REQ_DIMS:
        df[d] = to_numeric_1to5(df[d], context=f"REQ {os.path.basename(path)} / {d}")

    out = df.groupby("std_item")[REQ_DIMS].min().reset_index()
    return out


def load_doc_unified(path: str) -> pd.DataFrame:
    """
    Expect long format:
      dimension,score
    dimension must include the 4 DOC_DIMS.
    """
    df = read_csv_auto(path)
    df = clean_columns(df)

    # Allow aliases
    if "dimension" not in df.columns:
        for alias in ["指标", "整体指标", "文档整体指标"]:
            if alias in df.columns:
                df = df.rename(columns={alias: "dimension"})
                break
    if "score" not in df.columns:
        for alias in ["评分", "分数"]:
            if alias in df.columns:
                df = df.rename(columns={alias: "score"})
                break

    require_columns(df, ["dimension", "score"], context=f"DOC {os.path.basename(path)}")

    df["dimension"] = df["dimension"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    df["score"] = to_numeric_1to5(df["score"], context=f"DOC {os.path.basename(path)}")

    df = df[df["dimension"].isin(DOC_DIMS)].copy()

    missing = [d for d in DOC_DIMS if d not in set(df["dimension"])]
    if missing:
        raise KeyError(f"[DOC {os.path.basename(path)}] Missing dimensions {missing}. Found: {df['dimension'].tolist()}")

    return df[["dimension", "score"]]


# ----------------------------
# Core aggregations
# ----------------------------
def build_long_tables(req_wide: dict, doc_long_rows: list):
    """Build req_long and doc_long from loaded tables."""
    req_long_rows = []
    for rater, df in req_wide.items():
        for _, row in df.iterrows():
            item = row["std_item"]
            for d in REQ_DIMS:
                v = row[d]
                if pd.notna(v):
                    req_long_rows.append({
                        "rater": rater, "item": item,
                        "dimension": d, "score": float(v),
                        "level": "requirement",
                    })

    req_long = pd.DataFrame(req_long_rows)
    doc_long = pd.DataFrame(doc_long_rows)
    return req_long, doc_long


def consensus_table(df_long: pd.DataFrame) -> pd.DataFrame:
    """Per (item, dimension): median/mode/mode_freq/range/IQR/n_raters."""
    def _mode(s):
        m = s.mode()
        return m.iloc[0] if not m.empty else np.nan

    g = df_long.groupby(["item", "dimension"])["score"]
    out = pd.DataFrame({
        "consensus_median": g.median(),
        "consensus_mode": g.apply(_mode),
        "mode_freq": g.apply(lambda s: float((s == _mode(s)).mean()) if s.notna().any() else np.nan),
        "range": g.max() - g.min(),
        "IQR": g.quantile(0.75) - g.quantile(0.25),
        "n_raters": g.count(),
    }).reset_index()
    return out


def agreement_by_dimension(df_long: pd.DataFrame, k_total: int, require_all_raters: bool = False) -> pd.DataFrame:
    """
    Per dimension:
      exact_all_rate: rate of items where all observed scores equal (or strict: all raters present and equal)
      within1_rate: mean over items of fraction of raters within ±1 of item median
    """
    rows = []
    for dim, sub in df_long.groupby("dimension"):
        exact = []
        within1 = []
        for item, ssub in sub.groupby("item"):
            s = pd.to_numeric(ssub["score"], errors="coerce").dropna().to_numpy()

            if require_all_raters and len(s) != k_total:
                exact.append(0.0)
                within1.append(0.0)
                continue

            if len(s) == 0:
                exact.append(0.0)
                within1.append(0.0)
                continue

            exact.append(1.0 if np.max(s) == np.min(s) else 0.0)

            med = float(np.median(s))
            within1.append(float(np.mean(np.abs(s - med) <= 1)))

        rows.append({
            "dimension": dim,
            "exact_all_rate": float(np.mean(exact)) if exact else np.nan,
            "within1_rate": float(np.mean(within1)) if within1 else np.nan,
            "n_items": int(sub["item"].nunique()),
            "require_all_raters": bool(require_all_raters),
        })
    return pd.DataFrame(rows).sort_values(["dimension", "require_all_raters"])


def krippendorff_alpha_ordinal(matrix: np.ndarray, categories=(1, 2, 3, 4, 5)) -> float:
    """
    Krippendorff's alpha for ordinal data.
    matrix: shape (units, raters), np.nan allowed.
    """
    mat = np.array(matrix, dtype=float)
    cats = list(categories)
    k = len(cats)

    values = mat[~np.isnan(mat)]
    if values.size == 0:
        return np.nan

    freqs = np.array([np.sum(values == c) for c in cats], dtype=float)
    p = freqs / freqs.sum()

    # ordinal distance matrix
    delta = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            lo, hi = min(i, j), max(i, j)
            delta[i, j] = (p[lo:hi].sum()) ** 2

    # coincidence matrix
    O = np.zeros((k, k), dtype=float)
    for u in range(mat.shape[0]):
        unit = mat[u, :]
        unit = unit[~np.isnan(unit)]
        n_u = len(unit)
        if n_u < 2:
            continue
        counts = np.array([np.sum(unit == c) for c in cats], dtype=float)
        for i in range(k):
            for j in range(k):
                if i == j:
                    O[i, j] += counts[i] * (counts[i] - 1) / (n_u - 1)
                else:
                    O[i, j] += counts[i] * counts[j] / (n_u - 1)

    N = O.sum()
    if N == 0:
        return np.nan

    Do = float((O * delta).sum() / N)

    marg = O.sum(axis=1)
    E = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if i == j:
                E[i, j] = marg[i] * (marg[i] - 1) / (N - 1) if N > 1 else 0
            else:
                E[i, j] = marg[i] * marg[j] / (N - 1) if N > 1 else 0

    E_sum = E.sum()
    if E_sum == 0:
        return 1.0 if Do == 0 else np.nan

    De = float((E * delta).sum() / E_sum)
    if De == 0:
        return 1.0 if Do == 0 else np.nan

    return float(1 - Do / De)


def dist_to_median(df_long: pd.DataFrame) -> pd.DataFrame:
    """Per rater: mean abs deviation to per-cell median (MAE_to_median)."""
    med = df_long.groupby(["item", "dimension"])["score"].median().rename("median").reset_index()
    merged = df_long.merge(med, on=["item", "dimension"], how="left")
    merged["abs_dev"] = (merged["score"] - merged["median"]).abs()
    return merged.groupby("rater")["abs_dev"].mean().rename("MAE_to_median").reset_index()


def compute_bias(df_long: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    bias = mean(x - median) per rater (overall and by dimension).
    negative => stricter than consensus, positive => more lenient.
    """
    med = df_long.groupby(["item", "dimension"])["score"].median().rename("median").reset_index()
    merged = df_long.merge(med, on=["item", "dimension"], how="left")
    merged["delta"] = merged["score"] - merged["median"]

    overall = merged.groupby("rater")["delta"].mean().rename("bias_to_median").reset_index()

    by_dim = (
        merged.groupby(["dimension", "rater"])["delta"]
        .mean()
        .rename("bias_to_median")
        .reset_index()
        .sort_values(["dimension", "rater"])
    )
    return overall, by_dim


def run_friedman(req_long: pd.DataFrame, items_for_test=None) -> pd.DataFrame:
    rows = []
    for d in REQ_DIMS:
        mat = (
            req_long[req_long["dimension"] == d]
            .pivot_table(index="item", columns="rater", values="score", aggfunc="first")
        )
        if items_for_test is not None:
            mat = mat.reindex([i for i in items_for_test if i in mat.index])

        mat = mat.dropna(axis=0, how="any")
        n, k = mat.shape

        if n < 2 or k < 2:
            rows.append({"dimension": d, "n_items_used": n, "k_raters": k,
                         "friedman_chi2": np.nan, "p_value": np.nan, "kendall_W": np.nan})
            continue

        stat, p = friedmanchisquare(*[mat[c].to_numpy() for c in mat.columns])
        W = stat / (n * (k - 1)) if (n > 0 and k > 1) else np.nan
        rows.append({"dimension": d, "n_items_used": n, "k_raters": k,
                     "friedman_chi2": float(stat), "p_value": float(p), "kendall_W": float(W)})
    return pd.DataFrame(rows)


# ----------------------------
# Extended: trends / structure / outliers
# ----------------------------
def trend_by_dimension(cons: pd.DataFrame, low_threshold: float = 3.0) -> pd.DataFrame:
    """
    Summarize consensus trend per dimension:
      avg_median, pct_low (median<=threshold), avg_range, avg_IQR, avg_mode_freq
    """
    rows = []
    for dim, sub in cons.groupby("dimension"):
        rows.append({
            "dimension": dim,
            "n_items": int(sub["item"].nunique()),
            "avg_consensus_median": float(sub["consensus_median"].mean()),
            "median_of_consensus_median": float(sub["consensus_median"].median()),
            "pct_low_median(<=thr)": float((sub["consensus_median"] <= low_threshold).mean()),
            "avg_range": float(sub["range"].mean()),
            "avg_IQR": float(sub["IQR"].mean()),
            "avg_mode_freq": float(sub["mode_freq"].mean()),
        })
    return pd.DataFrame(rows).sort_values("dimension")


def trend_by_item(cons: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize consensus trend per item across dimensions:
      avg_median, min_median, worst_dim, avg_range, max_range_dim
    """
    rows = []
    for item, sub in cons.groupby("item"):
        # worst dimension by smallest consensus median; tie-break by largest range
        sub2 = sub.copy()
        sub2["tie"] = sub2["range"]
        worst = sub2.sort_values(["consensus_median", "tie"], ascending=[True, False]).iloc[0]
        maxr = sub2.sort_values(["range"], ascending=False).iloc[0]
        rows.append({
            "item": item,
            "avg_consensus_median": float(sub2["consensus_median"].mean()),
            "min_consensus_median": float(sub2["consensus_median"].min()),
            "worst_dimension": worst["dimension"],
            "worst_dimension_median": float(worst["consensus_median"]),
            "avg_range": float(sub2["range"].mean()),
            "max_range": float(maxr["range"]),
            "max_range_dimension": maxr["dimension"],
        })
    return pd.DataFrame(rows).sort_values(["avg_consensus_median", "min_consensus_median", "avg_range"])


def pairwise_distance_mae(df_long: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Pairwise distance between raters using MAE across common cells (item, dimension).
    Returns:
      - dist_matrix (square)
      - dist_long (rater_a, rater_b, n_cells, MAE_distance)
    """
    mat = df_long.pivot_table(index=["item", "dimension"], columns="rater", values="score", aggfunc="first")
    raters = list(mat.columns)
    dist = pd.DataFrame(index=raters, columns=raters, dtype=float)
    long_rows = []

    for a in raters:
        dist.loc[a, a] = 0.0

    for a, b in itertools.combinations(raters, 2):
        sub = mat[[a, b]].dropna(axis=0, how="any")
        n = int(sub.shape[0])
        if n == 0:
            d = np.nan
        else:
            d = float(np.mean(np.abs(sub[a].to_numpy() - sub[b].to_numpy())))
        dist.loc[a, b] = d
        dist.loc[b, a] = d
        long_rows.append({"rater_a": a, "rater_b": b, "n_cells": n, "MAE_distance": d})

    dist_long = pd.DataFrame(long_rows).sort_values("MAE_distance", ascending=False)
    return dist.reset_index().rename(columns={"index": "rater"}), dist_long


def avg_distance_from_others(dist_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Given square matrix with first col 'rater' and remaining raters as columns.
    Compute mean distance to others per rater.
    """
    df = dist_matrix.copy()
    raters = [c for c in df.columns if c != "rater"]
    out = []
    for _, row in df.iterrows():
        r = row["rater"]
        vals = [row[c] for c in raters if c != r]
        vals = [v for v in vals if pd.notna(v)]
        out.append({"rater": r, "avg_distance_to_others": float(np.mean(vals)) if vals else np.nan})
    return pd.DataFrame(out).sort_values("avg_distance_to_others", ascending=False)


def leave_one_out_disagreement(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Global disagreement G = mean over cells of (max-min across raters).
    For each rater r: compute G_{-r} after removing it.
    Contribution = G - G_{-r} (how much this rater inflates disagreement).
    """
    mat = df_long.pivot_table(index=["item", "dimension"], columns="rater", values="score", aggfunc="first")
    raters = list(mat.columns)

    def global_range(m):
        # per row range across available scores
        r = m.max(axis=1) - m.min(axis=1)
        return float(r.mean())

    G = global_range(mat)

    rows = []
    for r in raters:
        cols = [c for c in raters if c != r]
        m2 = mat[cols]
        G_minus = global_range(m2)
        rows.append({
            "rater": r,
            "G_all_mean_range": G,
            "G_minus_mean_range": G_minus,
            "contribution(G-G_minus)": float(G - G_minus),
        })
    return pd.DataFrame(rows).sort_values("contribution(G-G_minus)", ascending=False)


def rank_correlation_by_dimension(req_long: pd.DataFrame) -> pd.DataFrame:
    """
    For each dimension, compute pairwise rank correlation across items:
      Spearman rho and Kendall tau (based on item-wise scores).
    Output long table: dimension, rater_a, rater_b, n_items, spearman_rho, kendall_tau
    """
    rows = []
    for d in REQ_DIMS:
        mat = (
            req_long[req_long["dimension"] == d]
            .pivot_table(index="item", columns="rater", values="score", aggfunc="first")
        )
        raters = list(mat.columns)
        for a, b in itertools.combinations(raters, 2):
            sub = mat[[a, b]].dropna(axis=0, how="any")
            n = int(sub.shape[0])
            if n < 2:
                rho = np.nan
                tau = np.nan
            else:
                rho = float(spearmanr(sub[a].to_numpy(), sub[b].to_numpy()).correlation)
                tau = float(kendalltau(sub[a].to_numpy(), sub[b].to_numpy()).correlation)
            rows.append({
                "dimension": d, "rater_a": a, "rater_b": b,
                "n_items": n, "spearman_rho": rho, "kendall_tau": tau
            })
    return pd.DataFrame(rows).sort_values(["dimension", "spearman_rho"], ascending=[True, True])


def divergent_cells(df_long: pd.DataFrame, topk: int = 30, abs_threshold: float = 2.0) -> pd.DataFrame:
    """
    List cells where a rater deviates most from consensus median.
    Returns top deviations and counts per rater if needed.
    """
    med = df_long.groupby(["item", "dimension"])["score"].median().rename("median").reset_index()
    merged = df_long.merge(med, on=["item", "dimension"], how="left")
    merged["delta"] = merged["score"] - merged["median"]
    merged["abs_delta"] = merged["delta"].abs()

    # top deviations overall
    top = merged.sort_values(["abs_delta", "rater"], ascending=[False, True]).head(topk).copy()
    top = top[["rater", "item", "dimension", "score", "median", "delta", "abs_delta"]]

    # summary per rater
    summ = (
        merged.assign(is_hot=lambda x: x["abs_delta"] >= abs_threshold)
        .groupby("rater")
        .agg(
            n_cells=("abs_delta", "count"),
            hot_cells_ge_thr=("is_hot", "sum"),
            mean_abs_delta=("abs_delta", "mean"),
            pct_hot=("is_hot", "mean"),
        )
        .reset_index()
        .sort_values("mean_abs_delta", ascending=False)
    )
    return top, summ


def reliability_tables(req_long: pd.DataFrame, doc_matrix: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    reliability_req: per REQ dimension alpha + agreement rates (available/strict)
    reliability_doc: alpha on doc metrics as units
    """
    # agreement
    agree_avail = agreement_by_dimension(req_long, k_total=len(RATERS), require_all_raters=False)
    agree_strict = agreement_by_dimension(req_long, k_total=len(RATERS), require_all_raters=True)
    agree = pd.concat([agree_avail, agree_strict], ignore_index=True)

    # alpha per req dim
    rel_rows = []
    for d in REQ_DIMS:
        mat = (
            req_long[req_long["dimension"] == d]
            .pivot_table(index="item", columns="rater", values="score", aggfunc="first")
            .sort_index()
        )
        rel_rows.append({
            "dimension": d,
            "kripp_alpha_ordinal": krippendorff_alpha_ordinal(mat.to_numpy()),
        })
    rel_req = pd.DataFrame(rel_rows).merge(
        agree_avail.drop(columns=["require_all_raters"]).rename(columns={
            "exact_all_rate": "exact_all_rate_avail",
            "within1_rate": "within1_rate_avail",
            "n_items": "n_items_avail",
        }),
        on="dimension", how="left"
    ).merge(
        agree_strict.drop(columns=["require_all_raters"]).rename(columns={
            "exact_all_rate": "exact_all_rate_strict",
            "within1_rate": "within1_rate_strict",
            "n_items": "n_items_strict",
        }),
        on="dimension", how="left"
    )

    # doc alpha (4 units)
    # doc_matrix expected columns: dimension + raters...
    cols = [c for c in doc_matrix.columns if c != "dimension"]
    alpha_doc = krippendorff_alpha_ordinal(doc_matrix[cols].to_numpy())
    rel_doc = pd.DataFrame([{
        "units": "DOC metrics as units (4 dimensions)",
        "kripp_alpha_ordinal": alpha_doc,
        "note": "Only 4 units; interpret cautiously."
    }])

    return rel_req, rel_doc


# ----------------------------
# Main
# ----------------------------
def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- load all raters
    req_wide = {}
    doc_long_rows = []

    for rater, fns in RATERS.items():
        req_path = os.path.join(data_dir, fns["req"])
        doc_path = os.path.join(data_dir, fns["doc"])

        if not os.path.exists(req_path):
            raise FileNotFoundError(f"Missing file: {req_path}")
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Missing file: {doc_path}")

        req_wide[rater] = load_req_unified(req_path)
        doc_df = load_doc_unified(doc_path)

        for _, row in doc_df.iterrows():
            doc_long_rows.append({
                "rater": rater,
                "item": "DOC",
                "dimension": row["dimension"],
                "score": float(row["score"]),
                "level": "document",
            })

    # ---- build long tables
    req_long, doc_long = build_long_tables(req_wide, doc_long_rows)

    # ---- matrices
    req_matrix = (
        req_long
        .pivot_table(index=["item", "dimension"], columns="rater", values="score", aggfunc="first")
        .sort_index()
        .reset_index()
    )

    doc_matrix = (
        doc_long
        .pivot_table(index="dimension", columns="rater", values="score", aggfunc="first")
        .reindex(DOC_DIMS)
        .reset_index()
    )

    # ---- consensus + trends
    cons_req = consensus_table(req_long)
    cons_doc = consensus_table(doc_long)

    trend_dim_req = trend_by_dimension(cons_req, low_threshold=3.0)
    trend_item_req = trend_by_item(cons_req)

    trend_dim_doc = trend_by_dimension(cons_doc, low_threshold=3.0)  # dimension here is DOC_DIMS, item is DOC

    # ---- reliability
    rel_req, rel_doc = reliability_tables(req_long, doc_matrix)

    # ---- Friedman + W (use CORE if present)
    items_present = sorted(req_long["item"].dropna().unique().tolist())
    use_core = [i for i in CORE_ITEMS if i in items_present]
    items_for_test = use_core if len(use_core) >= 2 else None
    friedman_tbl = run_friedman(req_long, items_for_test)

    # ---- rater severity / closeness to consensus
    sev_req = req_long.groupby("rater")["score"].mean().rename("mean_score").reset_index()
    sev_doc = doc_long.groupby("rater")["score"].mean().rename("mean_score").reset_index()
    dist_req = dist_to_median(req_long)
    dist_doc = dist_to_median(doc_long)

    bias_req_overall, bias_req_by_dim = compute_bias(req_long)
    bias_doc_overall, bias_doc_by_dim = compute_bias(doc_long)

    rater_summary_req = (
        sev_req.merge(dist_req, on="rater", how="left")
        .merge(bias_req_overall, on="rater", how="left")
        .sort_values(["MAE_to_median", "abs_dev"] if "abs_dev" in sev_req.columns else ["MAE_to_median"])
    )
    rater_summary_doc = (
        sev_doc.merge(dist_doc, on="rater", how="left")
        .merge(bias_doc_overall, on="rater", how="left")
        .sort_values("MAE_to_median")
    )

    # ---- structure: rank correlation (req only)
    rankcorr_req = rank_correlation_by_dimension(req_long)

    # ---- outlier: pairwise distances + avg distance
    distmat_req, distlong_req = pairwise_distance_mae(req_long)
    avgdist_req = avg_distance_from_others(distmat_req)

    distmat_doc, distlong_doc = pairwise_distance_mae(doc_long)
    avgdist_doc = avg_distance_from_others(distmat_doc)

    # ---- outlier contribution (leave-one-out)
    loo_req = leave_one_out_disagreement(req_long)
    loo_doc = leave_one_out_disagreement(doc_long)

    # ---- divergent hotspots
    top_dev_req, dev_summ_req = divergent_cells(req_long, topk=40, abs_threshold=2.0)
    top_dev_doc, dev_summ_doc = divergent_cells(doc_long, topk=20, abs_threshold=2.0)

    # ---- export Excel
    out_xlsx = os.path.join(data_dir, "llm_rating_report.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # matrices
        req_matrix.to_excel(writer, sheet_name="req_matrix", index=False)
        doc_matrix.to_excel(writer, sheet_name="doc_matrix", index=False)

        # consensus + trends
        cons_req.to_excel(writer, sheet_name="consensus_req", index=False)
        cons_doc.to_excel(writer, sheet_name="consensus_doc", index=False)
        trend_dim_req.to_excel(writer, sheet_name="trend_dim_req", index=False)
        trend_item_req.to_excel(writer, sheet_name="trend_item_req", index=False)
        trend_dim_doc.to_excel(writer, sheet_name="trend_dim_doc", index=False)

        # reliability + friedman
        rel_req.to_excel(writer, sheet_name="reliability_req", index=False)
        rel_doc.to_excel(writer, sheet_name="reliability_doc", index=False)
        friedman_tbl.to_excel(writer, sheet_name="friedman_req", index=False)

        # rater summaries + bias
        rater_summary_req.to_excel(writer, sheet_name="rater_summary_req", index=False)
        rater_summary_doc.to_excel(writer, sheet_name="rater_summary_doc", index=False)
        bias_req_by_dim.to_excel(writer, sheet_name="bias_req_by_dim", index=False)
        bias_doc_by_dim.to_excel(writer, sheet_name="bias_doc_by_dim", index=False)

        # structure (rank correlations)
        rankcorr_req.to_excel(writer, sheet_name="rankcorr_req", index=False)

        # distances + outlier detection
        distmat_req.to_excel(writer, sheet_name="distmat_req", index=False)
        distlong_req.to_excel(writer, sheet_name="distpair_req", index=False)
        avgdist_req.to_excel(writer, sheet_name="avgdist_req", index=False)

        distmat_doc.to_excel(writer, sheet_name="distmat_doc", index=False)
        distlong_doc.to_excel(writer, sheet_name="distpair_doc", index=False)
        avgdist_doc.to_excel(writer, sheet_name="avgdist_doc", index=False)

        loo_req.to_excel(writer, sheet_name="loo_req", index=False)
        loo_doc.to_excel(writer, sheet_name="loo_doc", index=False)

        top_dev_req.to_excel(writer, sheet_name="top_dev_req", index=False)
        dev_summ_req.to_excel(writer, sheet_name="dev_summary_req", index=False)
        top_dev_doc.to_excel(writer, sheet_name="top_dev_doc", index=False)
        dev_summ_doc.to_excel(writer, sheet_name="dev_summary_doc", index=False)

    print("Saved report:", out_xlsx)
    print("Req items present:", items_present)
    if items_for_test is not None:
        print("Friedman used CORE items:", items_for_test)
    else:
        print("Friedman used ALL items (CORE not fully present).")


if __name__ == "__main__":
    main()
