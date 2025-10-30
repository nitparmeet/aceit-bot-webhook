# populate_cutoffs_from_table2.py
# ------------------------------------------------------------
# Build per-category closing ranks from "Table 2" and patch into
# the same workbook's "Cutoffs" sheet.
#
# Usage:
#   source venv/bin/activate
#   python3 populate_cutoffs_from_table2.py \
#       --file "Cutoffs MCC 2025 R1_August.xlsx" \
#       --year 2025 \
#       --round R1 \
#       --overwrite-cutoffs
#
# Output sheets created in the SAME workbook:
#   - Cutoffs_From_Table2            (normalized long table from Table 2)
#   - Cutoffs_Patched_From_Table2    (Cutoffs with ClosingRank filled/overwritten)
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


import pandas as pd
from unidecode import unidecode


# -----------------------
# Helpers (normalization)
# -----------------------

def _norm_text(s) -> str:
    if s is None:
        return ""
    s = unidecode(str(s))
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return s.strip()


def _name_key(name: str) -> str:
    s = _norm_text(name).upper()
    # Normalize GOVT ↔ GOVERNMENT
    s = re.sub(r"\bGOVT\b", "GOVERNMENT", s)
    # Drop trailing (code) like (123456)
    s = re.sub(r"\(\s*\d{4,6}\s*\)$", "", s)
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _canon_category_base(cat_raw: str) -> str:
    cat = _norm_text(cat_raw).upper()
    # Strip punctuation separators
    cat = re.sub(r"[^A-Z0-9 ]+", " ", cat)
    cat = re.sub(r"\s+", " ", cat).strip()

    # Base aliases
    if cat in {"UR", "GEN", "GENERAL", "OP", "OC", "OPEN"}:
        return "General"
    if cat in {"OBC", "BC"}:
        return "OBC"
    if cat == "EWS":
        return "EWS"
    if cat == "SC":
        return "SC"
    if cat == "ST":
        return "ST"

    # Token search
    toks = cat.split()
    for t in toks:
        if t in {"UR", "GEN", "GENERAL", "OP", "OC", "OPEN"}:
            return "General"
        if t in {"OBC", "BC"}:
            return "OBC"
        if t == "EWS":
            return "EWS"
        if t == "SC":
            return "SC"
        if t == "ST":
            return "ST"

    # Default conservative
    return "General"


def _detect_pwd(cat_raw: str, pwd_cell) -> bool:
    # PwD column wins if it says Y
    if pwd_cell is not None:
        pr = _norm_text(pwd_cell).upper()
        if pr in {"Y", "YES", "TRUE", "1"}:
            return True

    # Also honor PWD/PH inside category cell
    cat = _norm_text(cat_raw).upper()
    if re.search(r"\b(PWD|PH)\b", cat):
        return True

    return False


def _canon_quota_from_value(val: str) -> Optional[str]:
    """
    Normalize messy quota strings from Table 2 into internal buckets.

    Mappings used:
      - "Open Seat Quota", "All India Quota (AIQ)", "All India Quota" -> "AIQ"
      - "Central Institutions" -> "Central"
      - "Deemed/Paid Seats Quota" -> "Management"
      - "ESIC" -> "ESIC"
      - "AFMS"/"Armed Forces" -> "AFMS"
    """
    if val is None:
        return None
    t = _norm_text(val).upper()

    # Exact / contains checks for AIQ
    if t in {"ALL INDIA QUOTA (AIQ)", "ALL INDIA QUOTA", "AIQ", "OPEN SEAT QUOTA"}:
        return "AIQ"
    if ("ALL" in t and "INDIA" in t) or "AIQ" in t or "OPEN SEAT" in t:
        return "AIQ"

    if "CENTRAL" in t:
        return "Central"

    if ("DEEMED" in t) or ("PAID" in t) or ("MANAGEMENT" in t):
        return "Management"

    if "ESIC" in t or "ESI" in t:
        return "ESIC"

    if "AFMS" in t or "ARMED" in t or "FORCES" in t:
        return "AFMS"

    return None


def _round_tag_from_arg(year: int, round_tag: str) -> Tuple[int, str]:
    """
    Normalize round passed by user: 'R1', 'Round 1', '2025_R1' -> (year, 'R1')
    The CLI gives us a year explicitly; we return that year and a normalized 'R#'.
    """
    y = int(year)
    r = _norm_text(round_tag).upper()
    m = re.search(r"R(?:OUND)?\s*[-_ ]?\s*(\d+)", r)
    if m:
        return y, f"R{int(m.group(1))}"
    # Handle explicit '2025_R1' shape too
    m2 = re.search(r"_(R\d+)", r)
    if m2:
        return y, m2.group(1)
    # fallback: assume given is already like R1
    if re.fullmatch(r"R\d+", r):
        return y, r
    return y, "R1"


# -----------------------
# Core builders
# -----------------------

def build_long_from_table2(
    df: pd.DataFrame,
    year_filter: int,
    round_tag_filter: str
) -> pd.DataFrame:
    """
    Convert Table 2 to a normalized long table with:
      College Name, NameKey, QuotaCanon, CategoryBase, IsPwD, Year, RoundTag, ClosingRank
    Where ClosingRank is max(Rank) per group.
    """

    # Identify columns (lenient)
    cols_l = [c.lower() for c in df.columns]
    def pick(*cands):
        for cand in cands:
            for i, c in enumerate(cols_l):
                if cand == c or cand in c:
                    return df.columns[i]
        return None

    col_rank    = pick("rank")
    col_quota   = pick("quota")
    col_name    = pick("college name", "institute", "college")
    col_course  = pick("course", "program", "branch")
    col_pwd     = pick("pwd", "ph", "disab", "divyang")
    col_cat     = pick("category", "cat")

    needed = [col_rank, col_quota, col_name, col_course, col_cat]
    miss = [n for n in ["Rank", "Quota", "College Name", "Course", "Category"] if locals().get(f"col_{n.lower().replace(' ', '_')}") is None]
    if any(c is None for c in needed):
        raise ValueError(f"Table 2 is missing required columns. Found: {list(df.columns)}")

    # Keep MBBS only
    df = df.copy()
    if col_course and col_course in df.columns:
        df = df[df[col_course].astype(str).str.contains(r"\bMBBS\b", case=False, na=False)]

    # Basic normalization
    df["__College"] = df[col_name].map(_norm_text)
    df["__NameKey"] = df["__College"].map(_name_key)
    df["__QuotaCanon"] = df["Quota"].apply(_canon_quota_from_value)

    # Category / PwD
    df["__CategoryBase"] = df[col_cat].map(_canon_category_base)
    df["__IsPwD"] = df.apply(lambda r: _detect_pwd(r[col_cat], r[col_pwd] if col_pwd in df.columns else None), axis=1)

    # Year / Round
    y, rtag = _round_tag_from_arg(year_filter, round_tag_filter)
    df["__Year"] = y
    df["__Round"] = rtag

    # Rank → numeric (nullable Int64)
    # Convert to numeric; coerce bad values (like '—', '∞') to NaN
    rank = pd.to_numeric(df[col_rank], errors="coerce")

    # If ranks come as floats, floor them to whole numbers; leave NaN as-is
    # (You can use round() instead of floor() if you prefer)
    rank = np.floor(rank)

    # Keep as float for now (allows NaN to survive groupby), we'll finalize dtype later
    df["__Rank"] = pd.Series(rank, dtype="Float64")


    
    df["__QuotaCanon"] = df["Quota"].apply(_canon_quota_from_value)
    df["__CategoryBase"] = df["Category"].apply(_canon_category)
    
    

    # Drop rows without quota or rank
    df = df[df["__QuotaCanon"].notna() & df["__Rank"].notna()]

    # Closing rank = MAX rank per key
    grp = (
        df.groupby(["__NameKey", "__College", "__QuotaCanon", "__CategoryBase", "__IsPwD", "__Year", "__Round"], dropna=False)["__Rank"]
          .max()
          .reset_index()
    )

    out = grp.rename(columns={
        "__NameKey": "NameKey",
        "__College": "College Name",
        "__QuotaCanon": "QuotaCanon",
        "__CategoryBase": "CategoryBase",
        "__IsPwD": "IsPwD",
        "__Year": "Year",
        "__Round": "RoundTag",
        "__Rank": "ClosingRank",
    })

    # Ensure types
    
    # ClosingRank may be float (due to NaN propagation). Normalize to integer safely.
    out["ClosingRank"] = pd.to_numeric(out["ClosingRank"], errors="coerce")
    # use round() if you want .5 to become next integer; use floor() to be conservative:
    out["ClosingRank"] = np.floor(out["ClosingRank"])
    # final nullable integer dtype
    out["ClosingRank"] = out["ClosingRank"].astype("Int64")
    
    
    return out


def patch_cutoffs_in_same_workbook(
    book_path: Path,
    long_df: pd.DataFrame,
    overwrite: bool = False,
    target_sheet: str = "Cutoffs",
    out_sheet: str = "Cutoffs_Patched_From_Table2",
) -> None:
    """
    Patches the existing Cutoffs sheet (if present) inside the SAME workbook with
    ClosingRank computed from Table 2.

    Match keys:
      - NameKey (normalized college)
      - QuotaCanon
      - CategoryBase
      - IsPwD
      - Year
      - RoundTag
    """
    xls = pd.ExcelFile(book_path)

    # If there's no 'Cutoffs' sheet, just write long_df as-is to out_sheet and return
    if target_sheet not in xls.sheet_names:
        with pd.ExcelWriter(book_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            long_df.to_excel(w, sheet_name=out_sheet, index=False)
        print(f"[info] '{target_sheet}' not found. Wrote only '{out_sheet}'.")
        return

    base = pd.read_excel(book_path, sheet_name=target_sheet)

    # Identify base columns
    cols_l = [c.lower() for c in base.columns]
    def pick(*cands):
        for cand in cands:
            for i, c in enumerate(cols_l):
                if cand == c or cand in c:
                    return base.columns[i]
        return None

    col_name   = pick("college name", "institute", "college")
    col_quota  = pick("quota")
    col_round  = pick("round")
    col_cat    = pick("category", "cat")
    col_pwd    = pick("pwd")
    col_year   = pick("year", "session")
    col_close  = pick("closingrank", "closing rank", "rank")

    # Build keys in base
    base = base.copy()
    if col_name is None:
        raise ValueError(f"'{target_sheet}' must have a 'College Name' column (or similar). Found: {list(base.columns)}")

    base["NameKey"] = base[col_name].map(_name_key)

    # Quota canon in base (be lenient)
    if col_quota:
        base["QuotaCanon"] = base[col_quota].map(_canon_quota_from_value)
    else:
        base["QuotaCanon"] = None  # join will be looser if None

    # Category base in base
    if col_cat:
        base["CategoryBase"] = base[col_cat].map(_canon_category_base)
    else:
        base["CategoryBase"] = "General"

    # PwD in base
    if col_pwd:
        base["IsPwD"] = base[col_pwd].astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"])
    else:
        # also try to infer from Category containing PWD/PH
        base["IsPwD"] = base[col_cat].astype(str).str.upper().str.contains(r"\b(PWD|PH)\b", regex=True) if col_cat else False

    # Year
    if col_year and col_year in base.columns:
        yr = pd.to_numeric(base[col_year], errors="coerce")
        yr = np.floor(yr)  # or np.round(yr)
        base["Year"] = yr.astype("Int64")
    else:
        base["Year"] = pd.NA

    # Round tag
    if col_round:
        round_norm = base[col_round].astype(str).str.upper()
        # Try to extract 'R#' from things like '2025_R1', 'Round 1', 'R-1'
        tag = round_norm.str.extract(r"(R\s*[-_]?\s*\d+)", expand=False)
        tag = tag.str.replace(r"\s*[-_]\s*", "", regex=True).str.replace(" ", "", regex=False)  # 'R-1' -> 'R1'
        base["RoundTag"] = tag.fillna(round_norm)
    else:
        base["RoundTag"] = pd.NA

    # We will merge on these keys (nulls allowed, but better when present)
    merge_keys = ["NameKey", "QuotaCanon", "CategoryBase", "IsPwD", "Year", "RoundTag"]

    # Left-join long_df (from Table 2) onto base
    merged = base.merge(
        long_df[merge_keys + ["ClosingRank"]],
        on=merge_keys,
        how="left",
        suffixes=("", "_t2"),
    )

    # Decide how to fill ClosingRank
    if col_close is None:
        # If no closing column in base, create it
        col_close = "ClosingRank"

    if overwrite:
        merged[col_close] = merged["ClosingRank"]
    else:
        # fill only where missing
        if col_close in merged.columns:
            merged[col_close] = merged[col_close].where(merged[col_close].notna(), merged["ClosingRank"])
        else:
            merged[col_close] = merged["ClosingRank"]

    # Final write
    with pd.ExcelWriter(book_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        # Keep original columns order + ensure the closing rank column is present
        if col_close not in merged.columns:
            merged[col_close] = merged["ClosingRank"]
        merged.to_excel(w, sheet_name=out_sheet, index=False)
    print(f"[info] Wrote '{out_sheet}'.")


# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Workbook that has 'Table 2' and (optionally) 'Cutoffs'")
    ap.add_argument("--year", required=True, type=int, help="Year tag to assign (e.g., 2025)")
    ap.add_argument("--round", required=True, help="Round tag (e.g., R1)")
    ap.add_argument("--overwrite-cutoffs", action="store_true", help="Overwrite ClosingRank in Cutoffs; otherwise only fill blanks")
    args = ap.parse_args()

    book = Path(args.file)
    if not book.exists():
        raise FileNotFoundError(f"Excel not found: {book}")

    # Read Table 2
    xl = pd.ExcelFile(book)
    # Prefer a sheet named exactly "Table 2"; else pick first containing "table"
    if "Table 2" in xl.sheet_names:
        sheet_t2 = "Table 2"
    else:
        cands = [s for s in xl.sheet_names if "table" in s.lower()]
        sheet_t2 = cands[0] if cands else xl.sheet_names[0]

    t2 = pd.read_excel(book, sheet_name=sheet_t2)
    print(f"[info] Using Table 2 sheet: {sheet_t2}")
    print(f"[info] Table 2 columns: {list(t2.columns)}")

    # Build normalized long table
    long_df = build_long_from_table2(
        df=t2,
        year_filter=args.year,
        round_tag_filter=args.round
    )

    # Write normalized long into same workbook
    with pd.ExcelWriter(book, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        long_df.to_excel(w, sheet_name="Cutoffs_From_Table2", index=False)
    print("[info] Wrote normalized long table -> sheet 'Cutoffs_From_Table2'.")

    # Patch into Cutoffs
    patch_cutoffs_in_same_workbook(
        book_path=book,
        long_df=long_df,
        overwrite=args.overwrite_cutoffs,
        target_sheet="Cutoffs",
        out_sheet="Cutoffs_Patched_From_Table2",
    )
    print("[done] Wrote 'Cutoffs_Patched_From_Table2'.")


if __name__ == "__main__":
    main()
