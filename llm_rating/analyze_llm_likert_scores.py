# -*- coding: utf-8 -*-
"""
Unified-format analyzer for multi-LLM Likert scores.

Input (per rater):
1) Per-requirement CSV (wide):
   item,Unambiguous,Understandable,Correctness,Verifiable
2) Document-level CSV (long):
   dimension,score
   dimension ∈ {Internal Consistency, Non-redundancy, Completeness, Conciseness}

Output:
- llm_rating_report.xlsx (matrices + consensus + agreement + Krippendorff alpha + Friedman + summaries)

Dependencies:
  pip install pandas numpy scipy openpyxl
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare


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

# Optional: treat these as "core" requirements for some stats; if not found, script will use all items.
CORE_ITEMS = [f"R{i}" for i in range(1, 9)]  # R1..R8
EXT_ITEM = "C1"


# ----------------------------
# Utilities
# ----------------------------
def read_csv_auto(path: str) -> pd.DataFrame:
    """Try common encodings (UTF-8 with/without BOM, then GB encodings)."""
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
    if s in ["", "...", "…", "None", "nan"]:
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


# ----------------------------
# Load unified-format inputs
# ----------------------------
def load_req_unified(path: str) -> pd.DataFrame:
    """
    Expect wide format:
      item,Unambiguous,Understandable,Correctness,Verifiable
    Returns wide table with columns: std_item + REQ_DIMS
    """
    df = read_csv_auto(path)
    df = clean_columns(df)

    # Allow some aliases for the item column
    if "item" not in df.columns:
        for alias in ["需求条目", "需求编号", "功能点（对应原需求）", "功能点(对应原需求)"]:
            if alias in df.columns:
                df = df.rename(columns={alias: "item"})
                break

    require_columns(df, ["item"] + REQ_DIMS, context=f"REQ {os.path.basename(path)}")

    df["std_item"] = df["item"].apply(normalize_item_label)

    # If a rater splits items (e.g., R6a/R6b), aggregate back to std_item using MIN (weakest-link)
    out = df.groupby("std_item")[REQ_DIMS].min().reset_index()
    return out


def load_doc_unified(path: str) -> pd.DataFrame:
    """
    Expect long format:
      dimension,score
    dimension must include the 4 DOC_DIMS (order not required).
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

    # Normalize dimension strings
    df["dimension"] = df["dimension"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # Keep only the 4 doc dims if extra rows exist
    df = df[df["dimension"].isin(DOC_DIMS)].copy()

    # Validate completeness (optional: comment out if you allow missing)
    missing = [d for d in DOC_DIMS if d not in set(df["dimension"])]
    if missing:
        raise KeyError(f"[DOC {os.path.basename(path)}] Missing dimensions {missing}. Found: {df['dimension'].tolist()}")

    return df[["dimension", "score"]]


# ----------------------------
# Stats
# ----------------------------
def consensus_table(df_long: pd.DataFrame) -> pd.DataFrame:
    g = df_long.groupby(["item", "dimension"])["score"]
    return pd.DataFrame({
        "consensus_median": g.median(),
        "consensus_mode": g.agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
        "mode_freq": g.agg(lambda s: float((s == (s.mode().iloc[0] if not s.mode().empty else np.nan)).mean())
                           if not s.mode().empty else np.nan),
        "range": g.max() - g.min(),
        "IQR": g.quantile(0.75) - g.quantile(0.25),
        "n_raters": g.count(),
    }).reset_index()


def agreement_by_dimension(df_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim, sub in df_long.groupby("dimension"):
        exact_all = []
        within1 = []
        for _, ssub in sub.groupby("item"):
            scores = ssub["score"].to_numpy()
            exact_all.append(1 if len(set(scores)) == 1 else 0)
            med = float(np.median(scores))
            within1.append(float(np.mean(np.abs(scores - med) <= 1)))
        rows.append({
            "dimension": dim,
            "exact_all_rate": float(np.mean(exact_all)) if exact_all else np.nan,
            "within1_rate": float(np.mean(within1)) if within1 else np.nan,
            "n_items": int(sub["item"].nunique()),
        })
    return pd.DataFrame(rows).sort_values("dimension")


def krippendorff_alpha_ordinal(matrix: np.ndarray, categories=(1, 2, 3, 4, 5)) -> float:
    """
    Krippendorff's alpha for ordinal data.
    matrix: shape (units, raters), with np.nan allowed.
    """
    mat = np.array(matrix, dtype=float)
    cats = list(categories)
    k = len(cats)

    values = mat[~np.isnan(mat)]
    if values.size == 0:
        return np.nan

    # category probabilities
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
    med = df_long.groupby(["item", "dimension"])["score"].median().rename("median").reset_index()
    merged = df_long.merge(med, on=["item", "dimension"], how="left")
    merged["abs_dev"] = (merged["score"] - merged["median"]).abs()
    return merged.groupby("rater")["abs_dev"].mean().rename("MAE_to_median").reset_index()


def run_friedman(req_long: pd.DataFrame, items_for_test) -> pd.DataFrame:
    rows = []
    for d in REQ_DIMS:
        mat = (
            req_long[req_long["dimension"] == d]
            .pivot_table(index="item", columns="rater", values="score", aggfunc="first")
        )

        # restrict items if requested and present
        if items_for_test is not None:
            mat = mat.reindex([i for i in items_for_test if i in mat.index])

        # drop items with missing rater scores
        mat = mat.dropna(axis=0, how="any")

        n, k = mat.shape
        if n < 2 or k < 2:
            rows.append({"dimension": d, "n_items_used": n, "k_raters": k,
                         "friedman_chi2": np.nan, "p_value": np.nan, "kendall_W": np.nan})
            continue

        # If all raters give constant scores across items => no variance
        if mat.nunique().sum() == k:
            rows.append({"dimension": d, "n_items_used": n, "k_raters": k,
                         "friedman_chi2": np.nan, "p_value": np.nan, "kendall_W": np.nan})
            continue

        stat, p = friedmanchisquare(*[mat[c].to_numpy() for c in mat.columns])
        W = stat / (n * (k - 1)) if (n > 0 and k > 1) else np.nan
        rows.append({"dimension": d, "n_items_used": n, "k_raters": k,
                     "friedman_chi2": float(stat), "p_value": float(p), "kendall_W": float(W)})
    return pd.DataFrame(rows)


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
                "rater": rater, "item": "DOC",
                "dimension": row["dimension"],
                "score": float(row["score"]),
                "level": "document",
            })

    # ---- build req_long
    req_long_rows = []
    for rater, df in req_wide.items():
        for _, row in df.iterrows():
            item = row["std_item"]
            for d in REQ_DIMS:
                if pd.notna(row[d]):
                    req_long_rows.append({
                        "rater": rater, "item": item,
                        "dimension": d, "score": float(row[d]),
                        "level": "requirement",
                    })

    req_long = pd.DataFrame(req_long_rows)
    doc_long = pd.DataFrame(doc_long_rows)

    # ---- matrices
    req_matrix = (
        req_long
        .pivot_table(index=["item", "dimension"], columns="rater", values="score", aggfunc="first")
        .sort_index()
    )
    doc_matrix = (
        doc_long
        .pivot_table(index="dimension", columns="rater", values="score", aggfunc="first")
        .reindex(DOC_DIMS)
    )

    # ---- consensus & agreement
    cons_req = consensus_table(req_long)
    cons_doc = consensus_table(doc_long)

    agree_req = agreement_by_dimension(req_long)
    agree_doc = agreement_by_dimension(doc_long)

    # ---- Krippendorff alpha (ordinal)
    rel_req_rows = []
    for d in REQ_DIMS:
        mat = (
            req_long[req_long["dimension"] == d]
            .pivot_table(index="item", columns="rater", values="score", aggfunc="first")
            .sort_index()
        )
        rel_req_rows.append({
            "dimension": d,
            "kripp_alpha_ordinal": krippendorff_alpha_ordinal(mat.to_numpy()),
        })
    rel_req = pd.DataFrame(rel_req_rows).merge(agree_req, on="dimension", how="left")

    # Doc alpha: treat 4 doc metrics as 4 units
    doc_alpha = krippendorff_alpha_ordinal(doc_matrix.to_numpy())
    rel_doc = pd.DataFrame([{
        "units": "DOC metrics as units (4 dimensions)",
        "kripp_alpha_ordinal": doc_alpha,
        "note": "Only 4 units; interpret cautiously."
    }])

    # ---- Friedman + Kendall's W (use CORE_ITEMS if present, else all items)
    items_present = sorted(req_long["item"].dropna().unique().tolist())
    use_core = [i for i in CORE_ITEMS if i in items_present]
    items_for_test = use_core if len(use_core) >= 2 else None
    friedman_tbl = run_friedman(req_long, items_for_test)

    # ---- rater summaries: severity + MAE to median
    sev_req = req_long.groupby("rater")["score"].mean().rename("mean_score").reset_index()
    sev_doc = doc_long.groupby("rater")["score"].mean().rename("mean_score").reset_index()
    dist_req = dist_to_median(req_long)
    dist_doc = dist_to_median(doc_long)

    rater_summary_req = sev_req.merge(dist_req, on="rater", how="left").sort_values("MAE_to_median")
    rater_summary_doc = sev_doc.merge(dist_doc, on="rater", how="left").sort_values("MAE_to_median")

    # ---- export Excel
    out_xlsx = os.path.join(data_dir, "llm_rating_report_new.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        req_matrix.reset_index().to_excel(writer, sheet_name="req_matrix", index=False)
        doc_matrix.reset_index().to_excel(writer, sheet_name="doc_matrix", index=False)

        cons_req.to_excel(writer, sheet_name="consensus_req", index=False)
        cons_doc.to_excel(writer, sheet_name="consensus_doc", index=False)

        rel_req.to_excel(writer, sheet_name="reliability_req", index=False)
        rel_doc.to_excel(writer, sheet_name="reliability_doc", index=False)

        friedman_tbl.to_excel(writer, sheet_name="friedman_req", index=False)

        rater_summary_req.to_excel(writer, sheet_name="rater_summary_req", index=False)
        rater_summary_doc.to_excel(writer, sheet_name="rater_summary_doc", index=False)

    print("Saved report:", out_xlsx)
    print("Items present:", items_present)
    if items_for_test is not None:
        print("Friedman used CORE items:", items_for_test)
    else:
        print("Friedman used ALL items (CORE not fully present).")


if __name__ == "__main__":
    main()
