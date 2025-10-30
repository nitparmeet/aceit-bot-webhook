#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_unmatched_suggestions.py
-----------------------------
Create a "fill helper" CSV for unmatched cutoff rows by suggesting up to 3
best fuzzy matches from the target sheet.

Usage:
  python3 make_unmatched_suggestions.py \
    --unmatched unmatched_for_Cutoffs_FuzzyFilled.csv \
    --target-file "MCC_Final_with_Cutoffs_2024_2025.xlsx" \
    --target-sheet "Cutoffs" \
    --out unmatched_with_suggestions.csv
"""

import argparse
import re
from typing import List, Tuple

import pandas as pd
from unidecode import unidecode
from rapidfuzz import process, fuzz


# ----------------------------
# Helpers
# ----------------------------
def _norm_text(s) -> str:
    if s is None:
        return ""
    return unidecode(str(s)).strip()

def _name_key(name: str) -> str:
    s = _norm_text(name).upper()
    s = re.sub(r"\bGOVT\b", "GOVERNMENT", s)
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _safe_get(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for want in candidates:
        if want.lower() in cols:
            return cols[want.lower()]
    return None


# ----------------------------
# Core
# ----------------------------
def build_target_choices(tgt: pd.DataFrame) -> Tuple[List[str], pd.Series, pd.Series, pd.Series]:
    """
    Returns:
      choices: list[str] of "NAMEKEY || STATE || QUOTA || CATEGORY"
      key->row lookups (so we can return original fields)
    """
    col_name = _safe_get(tgt, ["College Name", "college", "institute"])
    col_state = _safe_get(tgt, ["State"])
    col_quota = _safe_get(tgt, ["Quota", "QuotaCanon", "__QuotaCanon"])
    col_cat   = _safe_get(tgt, ["Category", "CategoryBase", "__BaseCategory"])

    if not col_name:
        raise ValueError("Target sheet must have 'College Name' (or similar).")

    # Build a composite key for more reliable matching
    namekey = tgt[col_name].map(_name_key)
    state   = tgt[col_state] if col_state else ""
    quota   = tgt[col_quota] if col_quota else ""
    cat     = tgt[col_cat]   if col_cat   else ""

    composite = (namekey.fillna("") + " || " +
                 state.astype(str).fillna("") + " || " +
                 quota.astype(str).fillna("") + " || " +
                 cat.astype(str).fillna(""))

    # Return also columns we’ll want to copy alongside suggestions
    return composite.tolist(), tgt[col_name], state, quota


def suggest_for_row(row: dict,
                    choices: List[str],
                    tgt_name_col: pd.Series,
                    tgt_state_col: pd.Series,
                    tgt_quota_col: pd.Series,
                    n: int = 3) -> List[Tuple[str, int]]:
    """
    Make a query string from the source row and find top-N fuzzy matches
    over the prebuilt 'choices'.
    """
    src_name  = _name_key(row.get("College Name", ""))
    src_state = _norm_text(row.get("State", ""))
    src_quota = _norm_text(row.get("Quota", row.get("QuotaCanon", "")))
    src_cat   = _norm_text(row.get("Category", row.get("CategoryBase", "")))

    query = f"{src_name} || {src_state} || {src_quota} || {src_cat}"

    matches = process.extract(
        query, choices, scorer=fuzz.token_sort_ratio, limit=n
    )
    # matches -> list of tuples: (matched_string, score, index)
    out = []
    for m in matches:
        if not m:
            continue
        s, score, idx = m
        # Pull a friendlier display name from target
        disp = f"{tgt_name_col.iloc[idx]} | {str(tgt_state_col.iloc[idx])} | {str(tgt_quota_col.iloc[idx])}"
        out.append((disp, int(score)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unmatched", required=True, help="CSV produced by fuzzy_merge_cutoffs (unmatched_for_*.csv)")
    ap.add_argument("--target-file", required=True, help="Workbook that contains the main 'Cutoffs' sheet")
    ap.add_argument("--target-sheet", required=True, help="Sheet name to match against (e.g., 'Cutoffs')")
    ap.add_argument("--out", default="unmatched_with_suggestions.csv", help="Output CSV with suggestions")
    args = ap.parse_args()

    # Read inputs
    um = pd.read_csv(args.unmatched)
    tgt = pd.read_excel(args.target_file, sheet_name=args.target_sheet)

    # Build choice list from target
    choices, tgt_name, tgt_state, tgt_quota = build_target_choices(tgt)

    # Prepare output rows
    rows_out = []
    cand_cols = [
        "Sug1_Name", "Sug1_Score",
        "Sug2_Name", "Sug2_Score",
        "Sug3_Name", "Sug3_Score",
    ]

    # Normalize some column names in unmatched for safer access
    # (Keep originals too—just add if missing.)
    if "College Name" not in um.columns:
        alt = _safe_get(um, ["College Name", "college", "institute"])
        if alt and alt != "College Name":
            um["College Name"] = um[alt]

    if "Quota" not in um.columns and "QuotaCanon" in um.columns:
        um["Quota"] = um["QuotaCanon"]
    if "Category" not in um.columns and "CategoryBase" in um.columns:
        um["Category"] = um["CategoryBase"]

    for _, r in um.iterrows():
        row = r.to_dict()

        src_name = row.get("College Name", "")
        src_state = row.get("State", "")
        src_quota = row.get("Quota", row.get("QuotaCanon", ""))
        src_cat = row.get("Category", row.get("CategoryBase", ""))
        src_pwd = row.get("PwD (Y/N)", row.get("IsPwD", ""))
        src_close = row.get("ClosingRank", "")

        # Make suggestions
        suggestions = suggest_for_row(row, choices, tgt_name, tgt_state, tgt_quota, n=3)

        cand_vals = []
        for disp, score in suggestions:
            cand_vals.extend([disp, score])
        while len(cand_vals) < len(cand_cols):
            cand_vals.append("")

        # Build final row output
        row_out = {
            "Source_College": src_name,
            "Source_State": src_state,
            "Source_Quota": src_quota,
            "Source_Category": src_cat,
            "Source_PwD": src_pwd,
            "Source_ClosingRank": src_close,
        }
        row_out = {**row_out, **dict(zip(cand_cols, cand_vals))}
        rows_out.append(row_out)

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[done] Wrote suggestion file → {args.out}")


if __name__ == "__main__":
    main()
