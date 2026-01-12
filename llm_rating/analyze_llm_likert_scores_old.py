# -*- coding: utf-8 -*-
"""Analyze multi-LLM Likert scores for SRS quality metrics.

Inputs (place in the same directory or update paths):
- gpt_逐条需求评分.csv
- gpt_文档整体评分.csv
- deepseek_逐条需求评分表.csv
- deepseek_文档整体评分表.csv
- doubao_逐条需求评分表.csv
- doubao_文档整体评分表.csv
- qwen_逐条需求评分表.csv
- qwen_文档整体评分表.csv

Outputs:
- llm_rating_report.xlsx (unified matrices + consensus + reliability + Friedman + summaries)

Notes:
- Collapsing sub-items (e.g., R6a/R6b) uses MIN to preserve ordinal scale and reflect weakest-link quality.
"""

import os, re
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

DATA_DIR = os.path.dirname(__file__)
dims_req = ['Unambiguous', 'Understandable', 'Correctness', 'Verifiable']
core_items = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
doc_dims = ['Internal Consistency', 'Non-redundancy', 'Completeness', 'Conciseness']

def _is_number(x):
    try:
        return float(x) == float(x)
    except Exception:
        return False

def normalize_item_label(s: str):
    s = str(s).strip()
    if s in ["...", "…", ""]:
        return None
    if re.search(r"\bC1\b", s, re.I) or ("扩展" in s) or ("可扩展" in s):
        return "C1"
    if re.match(r"^\s*R9", s, re.I) or re.match(r"^\s*9", s):
        return "C1"
    m = re.match(r"^\s*(\d+)", s)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 8:
            return f"R{num}"
    m = re.match(r"^\s*R(\d+)", s, re.I)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 8:
            return f"R{num}"
        if num == 9:
            return "C1"
    if "C1" in s:
        return "C1"
    return s

def aggregate_by_item_min(df_wide, item_col="std_item"):
    return df_wide.groupby(item_col)[dims_req].min().reset_index()

def load_gpt_req(p):
    df = pd.read_csv(p).rename(columns={"功能点（对应原需求）": "item"})
    df["std_item"] = df["item"].apply(normalize_item_label)
    return aggregate_by_item_min(df[["std_item"] + dims_req])

def load_deepseek_req(p):
    df = pd.read_csv(p).rename(columns={"需求条目": "item"})
    df["std_item"] = df["item"].apply(normalize_item_label)
    return aggregate_by_item_min(df[["std_item"] + dims_req])

def load_qwen_req(p):
    df = pd.read_csv(p).rename(columns={"需求编号": "item"})
    mask = df[dims_req].apply(lambda col: col.map(_is_number)).all(axis=1)
    df = df.loc[mask, ["item"] + dims_req].copy()
    df["std_item"] = df["item"].apply(normalize_item_label)
    return aggregate_by_item_min(df[["std_item"] + dims_req])

def load_doubao_req(p):
    df = pd.read_csv(p)
    df = df[df["修订后编号"].astype(str).str.strip() != "..."].copy()
    dim_map = {
        "Unambiguous（无歧义）": "Unambiguous",
        "Understandable（易懂）": "Understandable",
        "Correctness（正确性）": "Correctness",
        "Verifiable（可验证）": "Verifiable",
    }
    df["dimension"] = df["指标"].map(lambda x: dim_map.get(str(x).strip(), str(x).strip()))
    df["std_item"] = df["修订后编号"].apply(normalize_item_label)
    wide = df.pivot_table(index="std_item", columns="dimension", values="评分", aggfunc="mean").reset_index()
    for d in dims_req:
        if d not in wide.columns:
            wide[d] = np.nan
    return aggregate_by_item_min(wide[["std_item"] + dims_req])

def load_gpt_doc(p):
    df = pd.read_csv(p)
    score_col = [c for c in df.columns if ("分数" in c) or (c == "评分")]
    score_col = score_col[0] if score_col else df.columns[1]
    return df.rename(columns={"指标":"dimension", score_col:"score"})[["dimension","score"]]

def load_deepseek_doc(p):
    return pd.read_csv(p).rename(columns={"文档整体指标":"dimension","评分":"score"})[["dimension","score"]]

def load_qwen_doc(p):
    return pd.read_csv(p).rename(columns={"指标":"dimension","评分":"score"})[["dimension","score"]]

def load_doubao_doc(p):
    df = pd.read_csv(p).rename(columns={"整体指标":"dimension","评分":"score"})[["dimension","score"]]
    mapping = {
        "Internal Consistency（内部一致性）": "Internal Consistency",
        "Non-redundancy（无冗余）": "Non-redundancy",
        "Completeness（完整性）": "Completeness",
        "Conciseness（简洁性）": "Conciseness",
    }
    df["dimension"] = df["dimension"].map(lambda x: mapping.get(str(x).strip(), str(x).strip()))
    return df

def consensus_table(df_long):
    g = df_long.groupby(["item","dimension"])["score"]
    return pd.DataFrame({
        "consensus_median": g.median(),
        "consensus_mode": g.agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
        "mode_freq": g.agg(lambda s: (s == s.mode().iloc[0]).mean() if not s.mode().empty else np.nan),
        "range": g.max() - g.min(),
        "IQR": g.quantile(0.75) - g.quantile(0.25),
        "n_raters": g.count(),
    }).reset_index()

def agreement_by_dimension(df_long):
    rows = []
    for dim, sub in df_long.groupby("dimension"):
        exact = []
        within1 = []
        for _, ssub in sub.groupby("item"):
            scores = ssub["score"].values
            exact.append(1 if len(set(scores)) == 1 else 0)
            med = np.median(scores)
            within1.append(np.mean(np.abs(scores - med) <= 1))
        rows.append({
            "dimension": dim,
            "exact_all_rate": float(np.mean(exact)) if exact else np.nan,
            "within1_rate": float(np.mean(within1)) if within1 else np.nan,
            "n_items": sub["item"].nunique(),
        })
    return pd.DataFrame(rows).sort_values("dimension")

def krippendorff_alpha_ordinal(matrix, categories=(1,2,3,4,5)):
    mat = np.array(matrix, dtype=float)
    cats = list(categories)
    k = len(cats)
    values = mat[~np.isnan(mat)]
    if len(values) == 0:
        return np.nan

    freqs = np.array([np.sum(values == c) for c in cats], dtype=float)
    p = freqs / freqs.sum()

    delta = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            lo, hi = min(i, j), max(i, j)
            delta[i, j] = (p[lo:hi].sum()) ** 2

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

    Do = (O * delta).sum() / N

    marg = O.sum(axis=1)
    E = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if i == j:
                E[i, j] = marg[i] * (marg[i] - 1) / (N - 1) if N > 1 else 0
            else:
                E[i, j] = marg[i] * marg[j] / (N - 1) if N > 1 else 0

    De = (E * delta).sum() / E.sum() if E.sum() != 0 else np.nan
    if De == 0 or np.isnan(De):
        return 1.0 if Do == 0 else np.nan
    return float(1 - Do / De)

def dist_to_median(df_long):
    med = df_long.groupby(["item","dimension"])["score"].median().rename("median").reset_index()
    merged = df_long.merge(med, on=["item","dimension"], how="left")
    merged["abs_dev"] = (merged["score"] - merged["median"]).abs()
    return merged.groupby("rater")["abs_dev"].mean().rename("MAE_to_median").reset_index()

def main():
    # Load requirement tables
    req_wide = {
        "gpt": load_gpt_req(os.path.join(DATA_DIR, "gpt_per-requirement_rating.csv")),
        "deepseek": load_deepseek_req(os.path.join(DATA_DIR, "deepseek_per-requirement_rating.csv")),
        "doubao": load_doubao_req(os.path.join(DATA_DIR, "doubao_per-requirement_rating.csv")),
        "qwen": load_qwen_req(os.path.join(DATA_DIR, "qwen_per-requirement_rating.csv")),
    }
    req_long = []
    for r, df in req_wide.items():
        for _, row in df.iterrows():
            for d in dims_req:
                if pd.notna(row[d]):
                    req_long.append({"rater": r, "item": row["std_item"], "dimension": d, "score": float(row[d])})
    req_long = pd.DataFrame(req_long)

    req_core = req_long[req_long["item"].isin(core_items)].copy()
    req_ext = req_long[req_long["item"] == "C1"].copy()

    core_matrix = req_core.pivot_table(index=["item","dimension"], columns="rater", values="score", aggfunc="first") \
                          .reindex(pd.MultiIndex.from_product([core_items, dims_req], names=["item","dimension"]))
    ext_matrix = req_ext.pivot_table(index=["item","dimension"], columns="rater", values="score", aggfunc="first")

    # Load doc tables
    doc_long = []
    doc_sources = {
        "gpt": load_gpt_doc(os.path.join(DATA_DIR, "gpt_document-wide_rating.csv")),
        "deepseek": load_deepseek_doc(os.path.join(DATA_DIR, "deepseek_document-wide_rating.csv")),
        "doubao": load_doubao_doc(os.path.join(DATA_DIR, "doubao_document-wide_rating.csv")),
        "qwen": load_qwen_doc(os.path.join(DATA_DIR, "qwen_document-wide_rating.csv")),
    }
    for r, df in doc_sources.items():
        for _, row in df.iterrows():
            doc_long.append({"rater": r, "item": "DOC", "dimension": row["dimension"], "score": float(row["score"])})
    doc_long = pd.DataFrame(doc_long)
    doc_matrix = doc_long.pivot_table(index="dimension", columns="rater", values="score", aggfunc="first").reindex(doc_dims)

    # Stats
    cons_core = consensus_table(req_core)
    cons_doc = consensus_table(doc_long)
    agree_core = agreement_by_dimension(req_core)
    agree_doc = agreement_by_dimension(doc_long)

    rel_core = []
    for d in dims_req:
        mat = req_core[req_core["dimension"] == d].pivot_table(index="item", columns="rater", values="score", aggfunc="first").reindex(core_items)
        rel_core.append({"dimension": d, "kripp_alpha_ordinal": krippendorff_alpha_ordinal(mat.values)})
    rel_core = pd.DataFrame(rel_core).merge(agree_core, on="dimension", how="left")

    doc_alpha = krippendorff_alpha_ordinal(doc_matrix.values)
    rel_doc = pd.DataFrame([{"units": "DOC metrics as units (4 dimensions)", "kripp_alpha_ordinal": doc_alpha}])

    friedman_rows = []
    for d in dims_req:
        mat = req_core[req_core["dimension"] == d].pivot_table(index="item", columns="rater", values="score", aggfunc="first").reindex(core_items)
        if mat.isna().any().any():
            continue
        if mat.nunique().sum() == mat.shape[1]:
            friedman_rows.append({"dimension": d, "friedman_chi2": np.nan, "p_value": np.nan, "kendall_W": np.nan, "n_items": mat.shape[0], "k_raters": mat.shape[1]})
            continue
        stat, p = friedmanchisquare(*[mat[c].values for c in mat.columns])
        W = stat / (mat.shape[0] * (mat.shape[1] - 1))
        friedman_rows.append({"dimension": d, "friedman_chi2": float(stat), "p_value": float(p), "kendall_W": float(W), "n_items": mat.shape[0], "k_raters": mat.shape[1]})
    friedman_tbl = pd.DataFrame(friedman_rows)

    dist_core = dist_to_median(req_core)
    dist_doc = dist_to_median(doc_long)
    sev_core = req_core.groupby("rater")["score"].mean().rename("mean_score").reset_index()
    sev_doc = doc_long.groupby("rater")["score"].mean().rename("mean_score").reset_index()

    # Export report
    out_xlsx = os.path.join(DATA_DIR, "llm_rating_report.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        core_matrix.reset_index().to_excel(writer, sheet_name="req_core_matrix", index=False)
        ext_matrix.reset_index().to_excel(writer, sheet_name="req_C1_matrix", index=False)
        doc_matrix.reset_index().to_excel(writer, sheet_name="doc_matrix", index=False)
        cons_core.to_excel(writer, sheet_name="consensus_req_core", index=False)
        cons_doc.to_excel(writer, sheet_name="consensus_doc", index=False)
        rel_core.to_excel(writer, sheet_name="reliability_req_core", index=False)
        rel_doc.to_excel(writer, sheet_name="reliability_doc", index=False)
        friedman_tbl.to_excel(writer, sheet_name="friedman_req_core", index=False)
        pd.merge(sev_core, dist_core, on="rater", how="left").to_excel(writer, sheet_name="rater_summary_req", index=False)
        pd.merge(sev_doc, dist_doc, on="rater", how="left").to_excel(writer, sheet_name="rater_summary_doc", index=False)

    print("Saved:", out_xlsx)

if __name__ == "__main__":
    main()
