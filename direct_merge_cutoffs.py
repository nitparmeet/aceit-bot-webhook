#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from typing import Optional, Tuple

import pandas as pd
from unidecode import unidecode
from rapidfuzz import process, fuzz


# ---------------------------
# Normalizers
# ---------------------------

def _t(x) -> str:
    if x is None:
        return ""
    return unidecode(str(x)).strip()

def _name_key(name: str) -> str:
    s = _t(name).upper()
    s = re.sub(r"\bGOVT\b", "GOVERNMENT", s)
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _canon_cat_base(cat: str) -> str:
    t = _t(cat).upper()
    if t in {"UR", "GEN", "GENERAL", "OPEN", "OP", "OC"}:
        return "GENERAL"
    if t in {"OBC", "BC"}:
        return "OBC"
    if t in {"EWS"}:
        return "EWS"
    if t in {"SC"}:
        return "SC"
    if t in {"ST"}:
        return "ST"
    return t or "GENERAL"

def _canon_pwd(v: str) -> str:
    t = _t(v).upper()
    if t in {"Y", "YES", "TRUE", "1", "PWD", "PH"}:
        return "Y"
    return "N"

def _canon_round_short(r: str) -> str:
    t = _t(r).upper()
    m = re.search(r"\bR(?:OUND)?\s*[- ]?(\d+)\b", t)
    if m:
        return f"R{int(m.group(1))}"
    m2 = re.search(r"_(R\d+)\b", t)
    if m2:
        return m2.group(1)
    if re.fullmatch(r"R\d+", t):
        return t
    return t

def _canon_quota_from_value(val: str) -> Optional[str]:
    if val is None:
        return None
    t = _t(val).upper()

    # AIQ
    if t in {"AIQ", "ALL INDIA QUOTA", "ALL INDIA", "15% AIQ", "15 AIQ", "15 PERCENT AIQ"}:
        return "AIQ"
    if ("ALL" in t and "INDIA" in t) or "AIQ" in t:
        return "AIQ"

    # Central
    if "CENTRAL" in t or t == "CU" or "CENTRAL INSTITUTION" in t or "CENTRAL UNIVERSITY" in t:
        return "Central"

    # Deemed / Management / Paid
    if ("DEEMED" in t) or ("MANAGEMENT" in t) or ("PAID" in t):
        return "Management"

    if "ESIC" in t or "ESI" in t:
        return "ESIC"

    if "AFMS" in t or "ARMED" in t or "FORCES" in t:
        return "AFMS"

    if "AIIMS" in t:
        return "AIIMS"

    if "JIPMER" in t:
        return "JIPMER"

    # State-ish
    if ("STATE" in t) or (t in {"SQ", "IP", "IPU", "STATEQUOTA", "DOMICILE", "HOMESTATE"}):
        return "State"

    return None

def _find_col(df: pd.DataFrame, want: list[str]) -> Optional[str]:
    cols = list(df.columns)
    lower = [unidecode(str(c)).strip().lower() for c in cols]
    for w in want:
        wnorm = w.strip().lower()
        for i, l in enumerate(lower):
            if wnorm == l:
                return cols[i]
    # contains fallback
    for w in want:
        for i, l in enumerate(lower):
            if w in l:
                return cols[i]
    return None


# ---------------------------
# Source (Cutoffs_Patched_From_Table2) loader
# ---------------------------

def read_source(path: str, sheet: str, year: int, round_tag: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)

    # Accept both our normalized sheet and “raw-like but already patched” shapes.
    # Try to detect minimal required columns; derive if needed.
    # Needed finally: College Name, QuotaCanon, CategoryBase, IsPwD, Year, RoundTag, ClosingRank

    # College Name
    if "College Name" not in df.columns:
        col_name = _find_col(df, ["college name", "college", "institute"])
        if not col_name:
            raise ValueError("Source sheet missing 'College Name'.")
        df["College Name"] = df[col_name]

    df["NameKey"] = df["College Name"].map(_name_key)

    # QuotaCanon
    if "QuotaCanon" not in df.columns:
        qcol = _find_col(df, ["quotacanon", "quota"])
        if qcol is None:
            raise ValueError("Source sheet missing 'Quota' / 'QuotaCanon'.")
        df["QuotaCanon"] = df[qcol].map(_canon_quota_from_value)

    # CategoryBase
    if "CategoryBase" not in df.columns:
        ccol = _find_col(df, ["categorybase", "category", "cat"])
        if ccol is None:
            raise ValueError("Source sheet missing 'Category' / 'CategoryBase'.")
        df["CategoryBase"] = df[ccol].map(_canon_cat_base)

    # IsPwD
    if "IsPwD" not in df.columns:
        pcol = _find_col(df, ["ispwd", "pwd", "ph"])
        if pcol is None and "Category" in df.columns:
            df["IsPwD"] = df["Category"].astype(str).str.contains(r"\b(PWD|PH)\b", case=False).map({True: "Y", False: "N"})
        elif pcol is not None:
            df["IsPwD"] = df[pcol].map(_canon_pwd)
        else:
            df["IsPwD"] = "N"

    # Year
    if "Year" not in df.columns:
        ycol = _find_col(df, ["year"])
        if ycol:
            df["Year"] = pd.to_numeric(df[ycol], errors="coerce").astype("Int64")
        else:
            df["Year"] = pd.Series([year] * len(df), dtype="Int64")

    # RoundTag
    if "RoundTag" not in df.columns:
        rcol = _find_col(df, ["roundtag", "round"])
        if rcol:
            df["RoundTag"] = df[rcol].map(_canon_round_short)
        else:
            df["RoundTag"] = _canon_round_short(round_tag)

    # ClosingRank
    if "ClosingRank" not in df.columns:
        ralias = _find_col(df, ["closing rank", "rank", "__rank"])
        if ralias:
            df["ClosingRank"] = pd.to_numeric(df[ralias], errors="coerce")
        else:
            raise ValueError("Source sheet has no ClosingRank / Rank.")

    # Keep only MBBS if a Course column exists
    ccol = _find_col(df, ["course"])
    if ccol:
        df = df[df[ccol].astype(str).str.contains(r"\bMBBS\b", case=False, na=False)]

    # Group to ClosingRank = max per key (protects against duplicates)
    df = (
        df.groupby(["NameKey", "College Name", "QuotaCanon", "CategoryBase", "IsPwD", "Year", "RoundTag"], dropna=False)["ClosingRank"]
        .max()
        .reset_index()
    )

    # Filter to the requested year only for primary pass; keep all for fallback
    df = df[df["Year"].astype("Int64") == int(year)].copy()

    return df


# ---------------------------
# Target (Cutoffs) loader
# ---------------------------

def read_target(path: str, sheet: str, year: int) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)

    for req in ["College Name", "Quota", "Category", "PwD (Y/N)", "Round"]:
        if req not in df.columns:
            raise ValueError(f"Target sheet '{sheet}' missing required column: {req}")

    df["NameKey"] = df["College Name"].map(_name_key)
    df["QuotaCanon"] = df["Quota"].map(_canon_quota_from_value)
    df["CategoryBase"] = df["Category"].map(_canon_cat_base)
    df["IsPwD"] = df["PwD (Y/N)"].map(_canon_pwd)
    df["RoundTag"] = df["Round"].map(_canon_round_short)

    # If target has Year column, filter to requested year to avoid mixing
    if "Year" in df.columns:
        df = df[df["Year"].astype(str).str.contains(str(year), na=False)]

    # Ensure ClosingRank exists & numeric
    if "ClosingRank" not in df.columns:
        df["ClosingRank"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df["ClosingRank"] = pd.to_numeric(df["ClosingRank"], errors="coerce").astype("Int64")

    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------
# Merge (exact then fuzzy per bucket)
# ---------------------------

def merge_cutoffs(source: pd.DataFrame, target: pd.DataFrame, fuzzy_threshold: int = 90) -> Tuple[pd.DataFrame, dict]:
    out = target.copy()

    # 1) Exact join on full key
    key_cols = ["NameKey", "QuotaCanon", "CategoryBase", "IsPwD", "RoundTag"]
    left = out.merge(
        source[key_cols + ["ClosingRank"]],
        on=key_cols,
        how="left",
        suffixes=("", "_SRC")
    )

    exact_hits = left["ClosingRank_SRC"].notna().sum()
    left.loc[left["ClosingRank_SRC"].notna(), "ClosingRank"] = left.loc[left["ClosingRank_SRC"].notna(), "ClosingRank_SRC"]
    left = left.drop(columns=["ClosingRank_SRC"])

    # 2) Fuzzy for remaining blanks, within same (QuotaCanon, CategoryBase, IsPwD, RoundTag) bucket
    still_blank = left["ClosingRank"].isna()
    filled_fuzzy = 0

    # Pre-bucket source choices
    src_by_bucket = {}
    for _, r in source.iterrows():
        b = (r["QuotaCanon"], r["CategoryBase"], r["IsPwD"], r["RoundTag"])
        src_by_bucket.setdefault(b, []).append((r["NameKey"], r["ClosingRank"]))

    for idx in left[still_blank].index:
        row = left.loc[idx]
        bucket = (row["QuotaCanon"], row["CategoryBase"], row["IsPwD"], row["RoundTag"])
        choices = src_by_bucket.get(bucket)
        if not choices:
            continue

        # match on NameKey; if None, skip
        name_key = row["NameKey"]
        cand_names = [c[0] for c in choices]
        match = process.extractOne(name_key, cand_names, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= fuzzy_threshold:
            best_name = match[0]
            # get rank for that best_name within bucket
            for nk, rk in choices:
                if nk == best_name and pd.notna(rk):
                    left.at[idx, "ClosingRank"] = int(rk)
                    filled_fuzzy += 1
                    break

    stats = {
        "exact_filled": int(exact_hits),
        "fuzzy_filled": int(filled_fuzzy),
        "remaining_blank": int(left["ClosingRank"].isna().sum()),
        "total_rows": int(len(left)),
    }
    return left, stats


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-file", required=True, help="Cutoffs MCC file, e.g. 'Cutoffs MCC 2025 R1_August.xlsx'")
    ap.add_argument("--source-sheet", default="Cutoffs_Patched_From_Table2", help="Sheet with patched ranks")
    ap.add_argument("--target-file", required=True, help="Your master workbook, e.g. 'MCC_Final_with_Cutoffs_2024_2025.xlsx'")
    ap.add_argument("--target-sheet", default="Cutoffs", help="Target sheet to update from")
    ap.add_argument("--year", type=int, required=True, help="Year filter (e.g., 2025)")
    ap.add_argument("--round", default="R1", help="Round tag to normalize (R1/R2/etc) — used only if needed for source")
    ap.add_argument("--out-sheet", default="Cutoffs_Merged_From_Table2", help="New sheet name to write into target workbook")
    ap.add_argument("--fuzzy-threshold", type=int, default=90, help="Fuzzy name match threshold (0–100)")
    args = ap.parse_args()

    print("[info] Reading source →", args.source_file)
    src = read_source(args.source_file, args.source_sheet, args.year, args.round)
    print("[info] Source rows:", len(src))

    print("[info] Reading target →", args.target_file, "/ sheet", repr(args.target_sheet))
    tgt = read_target(args.target_file, args.target_sheet, args.year)
    print("[info] Target rows:", len(tgt))

    merged, stats = merge_cutoffs(src, tgt, fuzzy_threshold=args.fuzzy_threshold)

    with pd.ExcelWriter(args.target_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
        merged.to_excel(xw, sheet_name=args.out_sheet, index=False)

    print(f"[done] Wrote merged sheet → '{args.out_sheet}' in {args.target_file}")
    print("[stats]", stats)


if __name__ == "__main__":
    main()
