# Full script: merge AIIMS seat counts into Sheet1 of College_Seat_New_Merged.xlsx
# No "melt" used. Robust header detection + fuzzy name matching.

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from unidecode import unidecode
from rapidfuzz import process, fuzz


# ---------- CONFIG ----------
COLLEGE_FILE = Path("College_Seat_New_Merged.xlsx")
AIIMS_FILE   = Path("FINAL SEAT MATRIX FOR AIIMS_BHU_JIMPER ROUND 1 UG 2025 (MBBS & BDS).xlsx")
OUT_FILE     = COLLEGE_FILE  # write back into same workbook adding a new sheet
OUT_SHEET    = "Sheet1_with_aiims_seats"
SOURCE_SHEET = "Sheet1"

# The seat columns we will populate in the output (overwrite if present)
TARGET_SEAT_COLS = [
    "EWS", "EWS PwD",
    "OBC", "OBC PwD",
    "SC",  "SC PwD",
    "ST",  "ST PwD",
    "TotalSeats",
]

# AIIMS header synonyms → our canonical seat columns
CAT_MAP: Dict[str, str] = {
    # UR appears in AIIMS, but you said not to carry a General column in the final mapping
    # We'll ignore UR/UR-PwD into final unless you want to add "General"/"General PwD".
    "ur": None, "gen": None, "open": None, "unreserved": None,
    "ur pwd": None, "ur-pwd": None, "gen pwd": None, "general pwd": None,

    "ews": "EWS",
    "ews pwd": "EWS PwD", "ews-pwd": "EWS PwD",

    "obc": "OBC",
    "obc pwd": "OBC PwD", "obc-pwd": "OBC PwD",

    "sc": "SC",
    "sc pwd": "SC PwD", "sc-pwd": "SC PwD",

    "st": "ST",
    "st pwd": "ST PwD", "st-pwd": "ST PwD",

    "total": "TotalSeats", "grand total": "TotalSeats", "total seats": "TotalSeats"
}

# minimal tokens to recognize a header row
HEADER_TOKENS = {"institute", "college", "name", "category", "seat", "seats", "total", "ews", "obc", "sc", "st", "pwd"}

# fuzzy match threshold
FUZZY_THRESHOLD = 92


# ---------- HELPERS ----------
def norm(s: str) -> str:
    if s is None:
        return ""
    s = unidecode(str(s)).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def build_key(name: str) -> str:
    return norm(name)


def read_college_sheet(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    # Ensure seat columns exist
    for c in TARGET_SEAT_COLS:
        if c not in df.columns:
            df[c] = 0
    if "College Name" not in df.columns:
        raise ValueError(f"'College Name' column is required in {sheet}")
    df["__key"] = df["College Name"].astype(str).map(build_key)
    return df


def find_aiims_header_row(raw: pd.DataFrame) -> int:
    """
    Scan rows to find header: the row that has an 'Institute' or 'College' like label
    and multiple recognizable category tokens.
    """
    # Work on a copy of text values
    arr = raw.fillna("").astype(str).values
    best_idx = -1
    best_score = -1

    for i in range(min(len(arr), 100)):
        row_vals = [unidecode(v).strip().lower() for v in arr[i]]
        row_text = " ".join(row_vals)
        score = 0

        # reward presence of institute/college/name
        if any(tok in row_text for tok in ("institute", "college", "name", "institute/college")):
            score += 2

        # count how many category tokens appear in row_vals
        tok_count = sum(1 for v in row_vals for t in ["ews", "obc", "sc", "st", "pwd", "total", "ur", "gen", "category"] if t in v)
        score += tok_count

        if score > best_score and score >= 3:
            best_score = score
            best_idx = i

    if best_idx == -1:
        # fallback: often the header is the first non-empty row below the title
        for i in range(min(len(arr), 50)):
            if any(v.strip() for v in arr[i]):
                return i
        return 0
    return best_idx


def read_aiims_raw(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet_name = xls.sheet_names[0]
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    # drop empty rows/cols
    raw = raw.dropna(how="all").reset_index(drop=True)
    raw = raw.loc[:, raw.notna().any(axis=0)]
    return raw


def canonicalize_aiims(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert weird AIIMS sheet into a clean wide table:
    columns: Institute, __key, EWS, EWS PwD, OBC, OBC PwD, SC, SC PwD, ST, ST PwD, TotalSeats
    """
    header_row = find_aiims_header_row(raw)
    df = pd.read_excel(AIIMS_FILE, sheet_name=0, header=header_row)
    # Drop fully empty cols
    df = df.loc[:, df.notna().any(axis=0)]

    # Standardize column names
    colmap = {}
    for c in df.columns:
        c_str = unidecode(str(c)).strip()
        colmap[c] = c_str
    df = df.rename(columns=colmap)

    # Find the institute column
    inst_col = None
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["institute", "name of institute", "college", "institute/college", "institute / college"]):
            inst_col = c
            break
    if inst_col is None:
        inst_col = df.columns[0]  # fallback to first column

    # Identify possible seat columns and map them to our target names
    used = {}
    for c in df.columns:
        lc = unidecode(str(c)).lower().strip()
        lc = lc.replace("—", "-").replace("–", "-")
        lc = re.sub(r"\s+", " ", lc)

        # normalize typical labels
        replacements = {
            "ur-pwd": "ur pwd",
            "ews-pwd": "ews pwd",
            "obc-pwd": "obc pwd",
            "sc-pwd": "sc pwd",
            "st-pwd": "st pwd",
            "general": "gen",
            "grand total": "total",
        }
        lc = replacements.get(lc, lc)

        target = CAT_MAP.get(lc)
        if target in TARGET_SEAT_COLS:
            used[target] = c  # remember source column

    # Build a compact frame with Institute and only the recognized seat columns
    cols_to_keep = [inst_col] + [used[t] for t in TARGET_SEAT_COLS if t in used]
    sub = df[cols_to_keep].copy()
    sub = sub.rename(columns={inst_col: "Institute", **{used[t]: t for t in used}})

    # Clean institute names
    sub["Institute"] = sub["Institute"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Coerce numeric and fill missing seat columns
    for t in TARGET_SEAT_COLS:
        if t not in sub.columns:
            sub[t] = 0
        sub[t] = pd.to_numeric(sub[t], errors="coerce").fillna(0).astype(int)

    # Some AIIMS sheets repeat headers mid-table. Drop rows where Institute equals header-like text
    bad_mask = sub["Institute"].str.lower().str.contains("institute|category|seat|mbbs|total", regex=True, na=False)
    sub = sub[~bad_mask].copy()
    sub = sub[sub["Institute"].str.len() > 0]

    # Aggregate by Institute (in case multiple rows)
    agg = sub.groupby("Institute", as_index=False).sum(numeric_only=True)

    # Build match key
    agg["__key"] = agg["Institute"].map(build_key)
    return agg[["Institute", "__key", *TARGET_SEAT_COLS]].copy()


def fuzzy_merge_into(col_df: pd.DataFrame, aiims_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fuzzy match aiims_df rows into col_df on __key/College Name.
    Overwrite seat columns when there's a good match.
    Return (updated_col_df, aiims_unmatched)
    """
    # Build a lookup of normalized college names from Sheet1
    name_map = dict(zip(col_df["__key"], col_df["College Name"]))

    # Prepare list for fuzzy choices (original names but keyed on normalized)
    sheet_keys = list(name_map.keys())

    col_df_updated = col_df.copy()
    matched_keys = set()
    unmatched = []

    for _, row in aiims_df.iterrows():
        k = row["__key"]
        if not k:
            unmatched.append(row)
            continue

        # exact first
        if k in name_map:
            best_key = k
            best_score = 100
        else:
            # fuzzy
            choice, best_score, _ = process.extractOne(
                query=k,
                choices=sheet_keys,
                scorer=fuzz.token_sort_ratio
            ) if sheet_keys else (None, 0, None)
            best_key = choice

        if best_key and best_score >= FUZZY_THRESHOLD:
            idx = col_df_updated.index[col_df_updated["__key"] == best_key]
            if len(idx) == 0:
                unmatched.append(row)
                continue
            # overwrite seats
            for c in TARGET_SEAT_COLS:
                if c in col_df_updated.columns:
                    col_df_updated.loc[idx, c] = int(row.get(c, 0))
            matched_keys.add(best_key)
        else:
            unmatched.append(row)

    aiims_unmatched = pd.DataFrame(unmatched) if unmatched else pd.DataFrame(columns=aiims_df.columns)
    return col_df_updated, aiims_unmatched


def main():
    print("Using:")
    print(f"  College file: {COLLEGE_FILE.resolve()}")
    print(f"  AIIMS file:   {AIIMS_FILE.resolve()}")
    print(f"  Out file:     {OUT_FILE.resolve()}")
    print()

    if not COLLEGE_FILE.exists():
        raise FileNotFoundError(f"{COLLEGE_FILE} not found")
    if not AIIMS_FILE.exists():
        raise FileNotFoundError(f"{AIIMS_FILE} not found")

    # Load college base
    col_df = read_college_sheet(COLLEGE_FILE, SOURCE_SHEET)

    # Read AIIMS raw and canonicalize
    raw = read_aiims_raw(AIIMS_FILE)
    aiims_df = canonicalize_aiims(raw)

    # Merge
    merged_df, aiims_unmatched = fuzzy_merge_into(col_df, aiims_df)

    # Write new sheet in same workbook
    with pd.ExcelWriter(OUT_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as xlw:
        merged_df.to_excel(xlw, sheet_name=OUT_SHEET, index=False)

    # Save unmatched for review
    aiims_unmatched.to_csv("AIIMS_unmatched.csv", index=False)

    print(f"Done. Wrote '{OUT_SHEET}' to {OUT_FILE.name}")
    print(f"Unmatched AIIMS rows: {len(aiims_unmatched)} → AIIMS_unmatched.csv")


if __name__ == "__main__":
    main()

