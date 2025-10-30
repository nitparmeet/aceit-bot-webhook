from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from unidecode import unidecode
from rapidfuzz import process, fuzz

# ---- Files/Sheets ----
COLLEGE_FILE = Path("College_Seat_New_Merged.xlsx")
AIIMS_FILE   = Path("FINAL SEAT MATRIX FOR AIIMS_BHU_JIMPER ROUND 1 UG 2025 (MBBS & BDS).xlsx")
SOURCE_SHEET = "Sheet1"
OUT_SHEET    = "Sheet1_with_aiims_seats"

# Your Sheet1 headers (we will write into these if they exist)
DEST_HEADERS = ["Gen","OBC","SC","ST","EWS","Total","EWS PwD","OBC PwD","SC PwD","ST PwD","TotalSeats"]

# Map many AIIMS/MCC labels -> destination headers
CAT_MAP = {
    # General
    "ur": "Gen", "general": "Gen", "gen": "Gen",
    "ur pwd": "Gen PwD", "general pwd": "Gen PwD", "gen pwd": "Gen PwD",
    # EWS
    "ews": "EWS", "ews pwd": "EWS PwD",
    # OBC
    "obc": "OBC", "obc pwd": "OBC PwD",
    # SC
    "sc": "SC", "sc pwd": "SC PwD",
    # ST
    "st": "ST", "st pwd": "ST PwD",
    # Total
    "total": "TotalSeats", "total seats": "TotalSeats", "grand total": "TotalSeats", "overall": "TotalSeats",
}

FUZZY_THRESHOLD = 80  # a bit looser to match AIIMS vs full names

def norm(s: str) -> str:
    if s is None: return ""
    s = unidecode(str(s)).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s

def keyify(name: str) -> str:
    return norm(name)

def read_sheet1(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SOURCE_SHEET)
    if "College Name" not in df.columns:
        raise ValueError(f"'College Name' column missing in {SOURCE_SHEET}")
    df["__key"] = df["College Name"].astype(str).map(keyify)
    # ensure destination headers exist
    for h in DEST_HEADERS:
        if h not in df.columns:
            df[h] = 0
    return df

def read_aiims_raw(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    # keep everything; don't drop columns here (merged titles can hide structure)
    raw = raw.fillna("")
    return raw

def guess_header_row(raw: pd.DataFrame) -> int:
    """
    Score rows by presence of header-like tokens; pick best.
    """
    tokens = ["institute", "college", "category", "ur", "gen", "ews", "obc", "sc", "st", "pwd", "total"]
    best_row, best_score = 0, -1
    for i in range(min(len(raw), 120)):
        row = " ".join(unidecode(str(x)).lower() for x in list(raw.iloc[i]))
        score = sum(t in row for t in tokens)
        if score > best_score:
            best_score, best_row = score, i
    return best_row

def normalize_headers(headers: List[str]) -> List[str]:
    out = []
    for h in headers:
        h2 = unidecode(str(h)).lower().strip()
        h2 = h2.replace("—","-").replace("–","-")
        h2 = re.sub(r"\s+", " ", h2)
        out.append(h2)
    return out

def pick_institute_and_seat_cols(df: pd.DataFrame) -> Tuple[str, List[str]]:
    """
    - Institute column: the text column with the *most non-empty, non-numeric strings*.
    - Seat columns: columns with mostly numeric values.
    """
    textiness = {}
    numericness = {}
    n = len(df)
    for c in df.columns:
        series = df[c]
        as_str = series.astype(str).str.strip()
        non_empty = (as_str != "")
        # "texty" = many non-empty and mostly not pure numbers
        is_number = pd.to_numeric(as_str, errors="coerce").notna()
        textiness[c] = (non_empty & ~is_number).sum()
        numericness[c] = is_number.sum()

    # choose institute col = max textiness
    inst_col = max(textiness, key=lambda k: textiness[k])

    # seat cols = columns with numeric majority (and not the institute col)
    seat_cols = [c for c in df.columns if c != inst_col and numericness[c] >= max(3, int(0.5*n))]

    return inst_col, seat_cols

def canonicalize_aiims(raw: pd.DataFrame) -> pd.DataFrame:
    hdr_row = guess_header_row(raw)
    df = raw.iloc[hdr_row+1:].reset_index(drop=True)  # data starts after header line
    df.columns = [f"col_{i}" for i in range(df.shape[1])]  # temporary names

    # Find institute + seat columns by content
    inst_col, seat_cols = pick_institute_and_seat_cols(df)

    # rename institute column
    df = df.rename(columns={inst_col: "Institute"})
    keep = ["Institute"] + seat_cols
    df = df[keep].copy()

    # drop header-like and blank-institute rows
    df["Institute"] = df["Institute"].astype(str).str.strip()
    df = df[df["Institute"] != ""]
    df = df[~df["Institute"].str.contains(r"\b(institute|category|mbbs|total)\b", case=False, regex=True)]

    # Attempt to read the *real* column headers from the header row:
    header_cells = list(raw.iloc[hdr_row])
    header_norm = normalize_headers(header_cells)
    # Build a mapping from our temporary seat_cols to detected headers (by position)
    # seat_cols are col_i; use their positional index to fetch header text
    col_index_map = {c: int(c.split("_")[1]) for c in seat_cols}
    seat_header_map: Dict[str,str] = {}
    for c, i in col_index_map.items():
        head = header_norm[i] if i < len(header_norm) else ""
        seat_header_map[c] = head

    # Map detected headers to destination names using CAT_MAP
    dst_col_map: Dict[str,str] = {}
    for c in seat_cols:
        h = seat_header_map.get(c,"")
        # normalize header token variants
        h = h.replace("ur-pwd","ur pwd").replace("ews-pwd","ews pwd").replace("obc-pwd","obc pwd").replace("sc-pwd","sc pwd").replace("st-pwd","st pwd")
        dest = CAT_MAP.get(h)
        if dest:
            dst_col_map[c] = dest

    # If we couldn't map by header text (merged or blank headers), fall back:
    # any seat columns we still don't recognize, we’ll try to infer later by column order.
    # For now keep them with their temp names; we’ll just ignore unmapped ones.

    # Coerce numeric
    for c in seat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    
    


    # Collapse duplicates by institute
    df = df.groupby("Institute", as_index=False).sum(numeric_only=True)

    # produce a tidy record with final destination columns
    out = pd.DataFrame({"Institute": df["Institute"]})
    # Initialize all destinations
    for d in DEST_HEADERS:
        out[d] = 0

    # Copy values
    for c in seat_cols:
        dest = dst_col_map.get(c)
        if not dest:
            continue
        out[dest] = df[c].values

    # Also support General/Gen Fill if present in seat headers but your Sheet1 has 'Gen'
    # (already handled via CAT_MAP → "Gen" or "Gen PwD")
    out["__key"] = out["Institute"].map(keyify)
    return out

def detect_dest_aliases(sheet_cols: List[str]) -> Dict[str,str]:
    """
    If your Sheet1 headers are exactly as you posted, we won’t need aliases.
    But we keep this to be safe.
    """
    aliases = {}
    # exacts first
    for h in DEST_HEADERS:
        if h in sheet_cols:
            aliases[h] = h
    # minimal regex fallback
    patt = {
        "Gen": [r"\bgen(eral)?\b"],
        "EWS": [r"\bews\b"],
        "OBC": [r"\bobc\b"],
        "SC":  [r"\bsc\b(?!.*pwd)"],
        "ST":  [r"\bst\b(?!.*pwd)"],
        "EWS PwD": [r"\bews.*pwd\b"],
        "OBC PwD": [r"\bobc.*pwd\b"],
        "SC PwD":  [r"\bsc.*pwd\b"],
        "ST PwD":  [r"\bst.*pwd\b"],
        "TotalSeats": [r"\btotal( seats?)?\b", r"\bgrand *total\b"],
    }
    for col in sheet_cols:
        n = unidecode(str(col)).lower().strip()
        for canon, pats in patt.items():
            if canon in aliases:  # already exact
                continue
            if any(re.search(p, n) for p in pats):
                aliases[canon] = col
    return aliases

def merge_into_sheet1(sheet1: pd.DataFrame, aiims: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dest_alias = detect_dest_aliases(list(sheet1.columns))
    updated = sheet1.copy()
    choices = list(updated["__key"])
    key2idx = {k:i for i,k in enumerate(choices)}

    unmatched_rows = []

    for _, r in aiims.iterrows():
        k = r["__key"]
        if not k or k == "central university":
            continue

        if k in key2idx:
            idx = key2idx[k]
            score = 100
        else:
            match = process.extractOne(k, choices, scorer=fuzz.token_sort_ratio)
            if not match or match[1] < FUZZY_THRESHOLD:
                unmatched_rows.append(r)
                continue
            idx = key2idx[match[0]]

        # Write into Sheet1 columns
        for d in DEST_HEADERS:
            colname = dest_alias.get(d, d)
            if colname in updated.columns and d in r.index:
                try:
                    updated.iat[idx, updated.columns.get_loc(colname)] = int(r[d])
                except Exception:
                    pass

    return updated, (pd.DataFrame(unmatched_rows) if unmatched_rows else pd.DataFrame(columns=aiims.columns))

def main():
    print("Loading Sheet1 …")
    sheet1 = read_sheet1(COLLEGE_FILE)
    print("Reading AIIMS file …")
    raw = read_aiims_raw(AIIMS_FILE)
    print("Canonicalizing AIIMS … (auto-detect headers & seats)")
    aiims = canonicalize_aiims(raw)

    print(f"AIIMS rows parsed: {len(aiims)}")
    if len(aiims) == 0:
        aiims.to_csv("AIIMS_unmatched.csv", index=False)
        raise SystemExit("Parsed 0 AIIMS rows — please open aiims_preview_top.csv to confirm header row visually.")

    print("Merging into Sheet1 …")
    merged, unmatched = merge_into_sheet1(sheet1, aiims)

    # Write back to same file as a new sheet
    with pd.ExcelWriter(COLLEGE_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
        merged.drop(columns=["__key"], errors="ignore").to_excel(xw, sheet_name=OUT_SHEET, index=False)

    unmatched.to_csv("AIIMS_unmatched.csv", index=False)
    print(f"Done. Wrote sheet '{OUT_SHEET}' inside {COLLEGE_FILE.name}")
    print(f"Unmatched rows: {len(unmatched)} → AIIMS_unmatched.csv")

if __name__ == "__main__":
    main()
