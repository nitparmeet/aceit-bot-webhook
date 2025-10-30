# -*- coding: utf-8 -*-
"""
cutoff_normalizer.py
--------------------
Final, robust normalizer + query helper for NEET-UG cutoff sheets.

- Detects header row automatically
- Standardizes columns: College Name, State, Course, Year, seat_type, Quota, Round, Category, PwD (Y/N), ClosingRank
- Normalizes Round (e.g., "2025_R1", "Round 1", "R-1") => R1 + Year
- Normalizes Quota ("AIQ", "All India Quota", "15% AIQ") => AIQ
- Normalizes Category (OP/UR/GEN => General; with/without PwD)
- Normalizes college names for reliable joins (upper, no punctuation, GOVT↔GOVERNMENT)
- Optional: writes a clean 'Cutoffs_Normalized' sheet you can use in your app

Usage:
  pip install pandas openpyxl Unidecode
  python3 cutoff_normalizer.py --file "YOUR_CUTOFFS.xlsx" --sheet "Cutoffs" \
      --round "2025_R1" --quota "AIQ" --category "General" --air 1 --write-normalized
"""

import argparse
import re
from typing import Tuple, Optional
import pandas as pd
from unidecode import unidecode

# ----------------------------
# Header detection & renaming
# ----------------------------
REQUIRED_HINTS = {
    "college": ["college", "institute", "inst"],
    "category": ["category", "cat"],
    "round": ["round"],
    "quota": ["quota"],
    "rank": ["closing", "closingrank", "closing rank", "closing_rank", "closing rank (air)", "air"],
}

CANONICAL_COLS = {
    "College Name": ["college", "institute", "inst name", "institute name"],
    "State": ["state"],
    "Course": ["course", "branch", "program"],
    "Year": ["year", "session"],
    "seat_type": ["seat type", "seat_type", "seat category", "seat cat"],
    "Quota": ["quota", "all india quota", "aiq", "state quota", "deemed", "central"],
    "Round": ["round", "rnd"],
    "Category": ["category", "cat"],
    "PwD (Y/N)": ["pwd", "ph", "pwd (y/n)", "is_pwd", "is ph", "ph (y/n)"],
    "ClosingRank": ["closing rank", "closingrank", "rank", "closing rank (air)"],
    "Source": ["source"],
    "Notes": ["notes", "remark", "remarks"],
}

def find_header_row(df: pd.DataFrame, search_rows: int = 25) -> int:
    """
    Find the header row by looking for a row that contains hints for required fields.
    Returns 0 if not found (assumes first row is header).
    """
    best_row = 0
    best_score = -1
    for i in range(min(search_rows, len(df))):
        vals = df.iloc[i].astype(str).str.lower().tolist()
        score = 0
        for hints in REQUIRED_HINTS.values():
            if any(any(h in v for h in hints) for v in vals):
                score += 1
        if score > best_score:
            best_score = score
            best_row = i
    return best_row

def pick_col_name_map(cols) -> dict:
    """
    Map messy column names to canonical names.
    """
    mapping = {}
    for c in cols:
        cl = unidecode(str(c)).strip().lower()
        for canon, hints in CANONICAL_COLS.items():
            for h in hints:
                if h in cl:
                    mapping[c] = canon
                    break
            if c in mapping:
                break
    return mapping

# ----------------------------
# Normalizers
# ----------------------------
GOVT_EQUIV = [(r"\bGOVT\b", "GOVERNMENT")]

def norm_text(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = unidecode(str(s))
    s = s.replace("’","'").replace("“","\"").replace("”","\"")
    return s.strip()

def norm_name_key(name: str) -> str:
    """Stable key for college names (UPPER, strip punctuation, GOVT→GOVERNMENT, compact spaces)."""
    if name is None:
        return ""
    s = str(name).upper()
    s = re.sub(r"\bGOVT\b", "GOVERNMENT", s)
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_round(value: str) -> Tuple[Optional[int], Optional[int], str]:
    """
    Returns (year, round_num, canon_round_str)
      - Accepts "2025_R1", "R1", "Round 1", "R-1", "UG 2025 Round 1"
    canon_round_str: "R1", "R2", "R3", "MOPUP", "STRAY" (best effort)
    """
    v = norm_text(value).upper()
    # Look for year like 2024/2025
    year = None
    ym = re.search(r"(20\d{2})", v)
    if ym:
        year = int(ym.group(1))

    # Round types
    if re.search(r"STRAY", v):
        return (year, None, "STRAY")
    if re.search(r"MOP[-\s]*UP|MOPUP", v):
        return (year, None, "MOPUP")

    m = re.search(r"R(?:OUND)?\s*[-_ ]?\s*(\d+)", v)
    if m:
        r = int(m.group(1))
        return (year, r, f"R{r}")

    # Special form like "2025_R1"
    m2 = re.search(r"_(R\d+)\b", v)
    if m2:
        cr = m2.group(1)
        mnum = re.search(r"R(\d+)", cr)
        r = int(mnum.group(1)) if mnum else None
        return (year, r, cr)

    # Fallback: empty
    return (year, None, "")

QUOTA_CANON = {
    "AIQ": ["AIQ", "ALL INDIA QUOTA", "ALL INDIA", "15% AIQ", "AIQ (15%)", "CENTRAL AIQ"],
    "DEEMED": ["DEEMED", "DEEMED UNIVERSITY"],
    "CENTRAL": ["CENTRAL", "CENTRAL UNIVERSITY", "CU"],
    "ESIC": ["ESIC", "ESI"],
    "AFMS": ["AFMS", "ARMED FORCES"],
    "STATE": ["STATE", "STATE QUOTA", "SQ"],
}

def canon_quota(q: str) -> str:
    v = norm_text(q).upper()
    for key, alts in QUOTA_CANON.items():
        if v == key:
            return key
        for a in alts:
            if v == a:
                return key
    # fuzzy contains
    if "ALL INDIA" in v or "AIQ" in v:
        return "AIQ"
    if "DEEMED" in v:
        return "DEEMED"
    if "CENTRAL" in v or "CU" == v:
        return "CENTRAL"
    if "ESIC" in v or "ESI" in v:
        return "ESIC"
    if "AFMS" in v or "ARMED FORCES" in v:
        return "AFMS"
    if "STATE" in v or v == "SQ":
        return "STATE"
    return v  # return as-is if unknown

CAT_BASES = {
    "GENERAL": ["GEN", "GENERAL", "UR", "OP", "OC", "OPEN"],
    "EWS": ["EWS"],
    "OBC": ["OBC", "BC"],
    "SC": ["SC"],
    "ST": ["ST"],
}

def split_category(cat_raw: str, pwd_raw: Optional[str]) -> Tuple[str, bool]:
    """
    Derive (base_category, is_pwd)
      - understands combined tokens like "OP PH", "UR-PwD", "GEN PWD", "OBC-PH"
      - pwd_raw (Y/N) column can force PwD flag
    """
    cat = norm_text(cat_raw).upper()
    is_pwd = False

    # If separate PwD column says yes
    if pwd_raw is not None:
        pr = norm_text(pwd_raw).upper()
        if pr in {"Y", "YES", "TRUE", "1"}:
            is_pwd = True

    # Detect PwD in category cell as well
    if re.search(r"\b(PWD|PH)\b", cat):
        is_pwd = True
        cat = re.sub(r"\b(PWD|PH)\b", "", cat).strip()

    # Normalize separators
    cat = re.sub(r"[^A-Z0-9 ]+", " ", cat)
    cat = re.sub(r"\s+", " ", cat).strip()

    # Match base
    base = ""
    for k, alts in CAT_BASES.items():
        if cat == k:
            base = k
            break
        for a in alts:
            if cat == a or re.search(rf"\b{a}\b", cat):
                base = k
                break
        if base:
            break

    # Fallback: try to guess by tokens
    if not base:
        tokens = cat.split()
        for t in tokens:
            for k, alts in CAT_BASES.items():
                if t == k or t in alts:
                    base = k
                    break
            if base:
                break

    # Default to GENERAL if still blank (conservative choice)
    if not base:
        base = "GENERAL"

    return base, is_pwd

# ----------------------------
# Load, normalize, filter
def load_and_normalize(path: str, sheet: str | None):
    """
    Load Excel, auto-detect sheet if not provided or wrong, detect header row,
    and produce a normalized dataframe + header_row index + column mapping.
    """
    import pandas as pd
    from unidecode import unidecode

    def _auto_pick_sheet(xlsx_path: str) -> str:
        xl = pd.ExcelFile(xlsx_path)
        cand = [s for s in xl.sheet_names]
        # prefer sheets that look like cutoffs/seat matrices
        priority_keywords = ["cutoff", "cut-offs", "closing", "seat", "matrix", "aiims", "bhu", "jipmer", "round"]
        scored = []
        for s in cand:
            score = 0
            name_l = s.lower()
            score += sum(k in name_l for k in priority_keywords)
            try:
                peek = pd.read_excel(xlsx_path, sheet_name=s, header=None, nrows=30)
                text = " ".join(str(x) for x in peek.fillna("").values.ravel()).lower()
                score += 2 * sum(k in text for k in priority_keywords)
            except Exception:
                pass
            scored.append((score, s))
        scored.sort(reverse=True)
        return scored[0][1] if scored else xl.sheet_names[0]

    # If a sheet name was supplied but isn't present, fall back automatically
    xl = pd.ExcelFile(path)
    actual_sheet = sheet
    if actual_sheet not in xl.sheet_names:
        actual_sheet = _auto_pick_sheet(path)
        print(f"[info] Worksheet '{sheet}' not found. Using auto-detected sheet: '{actual_sheet}'")

    # Read raw
    raw = pd.read_excel(path, sheet_name=actual_sheet, header=None)

    # ---- detect header row: first row containing any of these header tokens ----
    header_tokens = {
        "college", "institute", "name", "code",
        "category", "quota", "round", "closing", "rank", "pwd", "seat",
        "state", "course", "year", "allotted"
    }
    header_row = 0
    for i in range(min(50, len(raw))):
        row_text = " ".join(str(x) for x in raw.iloc[i].fillna("")).lower()
        hits = sum(1 for t in header_tokens if t in row_text)
        if hits >= 3:  # looks like header
            header_row = i
            break

    df = pd.read_excel(path, sheet_name=actual_sheet, header=header_row)
    # normalize columns
    df.columns = [unidecode(str(c)).strip() for c in df.columns]

    # build a loose column map
    colmap = {}
    cols_l = [c.lower() for c in df.columns]

    def _pick(*cands):
        for cand in cands:
            for i, c in enumerate(cols_l):
                if cand in c:
                    return df.columns[i]
        return None

    colmap["college_name"] = _pick("college", "institute", "name")
    colmap["college_code"] = _pick("code")
    colmap["state"]       = _pick("state")
    colmap["course"]      = _pick("course", "mbbs", "bds")
    colmap["year"]        = _pick("year", "session")
    colmap["quota"]       = _pick("quota", "category of seats", "aiq")
    colmap["round"]       = _pick("round")
    colmap["category"]    = _pick("category", "cat")
    colmap["pwd"]         = _pick("pwd", "disab", "ph", "divyang")
    colmap["closingrank"] = _pick("closing", "closing rank", "closing_rank", "cutoff", "closingrank", "last rank")

    # Informative printout for debugging
    print("[info] Using sheet:", actual_sheet)
    print("[info] Header row:", header_row)
    print("[info] Columns:", list(df.columns))
    print("[info] Column map:", colmap)

    # light cleaning of key columns
    def _norm_text(s):
        if pd.isna(s): return ""
        return unidecode(str(s)).strip()

    for k in ["college_name", "quota", "category", "pwd", "round", "course", "state"]:
        if colmap.get(k) in df.columns:
            df[colmap[k]] = df[colmap[k]].map(_norm_text)

    # canonicalize some key values
    def canon_category(x: str) -> str:
        x = (x or "").upper().replace(" ", "")
        # common aliases
        mapping = {
            "OP": "General", "UR": "General", "GEN": "General", "GENERAL": "General",
            "OBC": "OBC", "BC": "OBC",
            "EWS": "EWS",
            "SC": "SC",
            "ST": "ST",
        }
        # leave PWD to a separate flag
        return mapping.get(x, (x or "").title())

    def canon_pwd(x: str) -> str:
        t = (x or "").strip().lower()
        if t in {"y", "yes", "pwd", "pd", "true", "1"}: return "Y"
        if t in {"n", "no", "false", "0"}: return "N"
        # if category contains 'pwd' we’ll set later in your downstream logic
        return "N"

    if colmap.get("category"):
        df[colmap["category"]] = df[colmap["category"]].apply(canon_category)
    if colmap.get("pwd"):
        df[colmap["pwd"]] = df[colmap["pwd"]].apply(canon_pwd)

    return df, header_row, colmap


def filter_cutoffs(df: pd.DataFrame, want_round: str, want_quota: str, want_category: str, air: int) -> pd.DataFrame:
    """
    Filter normalized DF using robust matches.
    want_round: accepts "2025_R1" / "R1" / "Round 1" / "R-1"
    want_quota: e.g., "AIQ" (synonyms handled)
    want_category: "General" / "EWS" / "OBC" / "SC" / "ST"
    """
    yr, rnum, canon = parse_round(want_round)
    cq = canon_quota(want_quota)
    base, want_pwd = split_category(want_category, None)  # If they pass 'General PwD' it will set pwd=True

    # Allow both explicit "R1" canonical and numeric equality
    mask_round = False
    if canon:
        mask_round = (df["__CanonRound"] == canon)
    if rnum is not None:
        mask_round = mask_round | (df["__RoundNum"] == rnum)

    # If year was provided, respect it; else ignore
    if yr:
        mask_round = mask_round & ((df["__Year"] == yr) | (df["__Year"].isna()))

    mask = (
        mask_round &
        (df["__CanonQuota"] == cq) &
        (df["__BaseCategory"] == base) &
        # If user asked PwD explicitly in category, enforce it; else accept both PwD and non-PwD
        ((df["__IsPwD"] == want_pwd) if want_pwd else (df["__IsPwD"].isin([False, None])))
    )

    # AIR eligibility
    if air is not None:
        mask = mask & (df["__ClosingRankNum"].notna()) & (air <= df["__ClosingRankNum"])

    return df[mask].copy()

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Excel file containing Cutoffs")
    ap.add_argument("--sheet", default=None, help="Sheet name (default: auto / first)")
    ap.add_argument("--round", required=True, help='Round value like "2025_R1", "R1", "Round 1"')
    ap.add_argument("--quota", required=True, help='Quota like "AIQ", "State", "Deemed", "Central"')
    ap.add_argument("--category", required=True, help='Category like "General", "EWS", "OBC", "SC", "ST" (add " PwD" if needed)')
    ap.add_argument("--air", type=int, required=True, help="All India Rank to test eligibility")
    ap.add_argument("--write-normalized", action="store_true", help="Write Cutoffs_Normalized sheet back into the file")
    args = ap.parse_args()

    df, header_row, colmap = load_and_normalize(args.file, args.sheet)

    # Diagnostics before filtering
    print("=== HEADER DETECT ===")
    print(f"Detected header row: {header_row}")
    print("Column mapping → canonical:")
    for k, v in colmap.items():
        print(f"  {k!r}  →  {v}")

    # Round / Quota / Category shapes
    yr, rnum, canon = parse_round(args.round)
    cq = canon_quota(args.quota)
    base, want_pwd = split_category(args.category, None)
    # counts
    cnt_round = ((df["__CanonRound"] == canon) | ((rnum is not None) & (df["__RoundNum"] == rnum))).sum()
    cnt_quota = (df["__CanonQuota"] == cq).sum()
    cnt_cat = (df["__BaseCategory"] == base).sum()

    print("\n=== DIAGNOSTICS ===")
    print(f"Your input → Round: {args.round} (norm: {canon or ''} num={rnum} year={yr}), Quota: {args.quota} (norm: {cq}), Category: {args.category} (base: {base} PwD={want_pwd})")
    print(f"Rows matching round: {cnt_round}")
    print(f"Rows matching quota: {cnt_quota}")
    print(f"Rows matching category: {cnt_cat}")

    out = filter_cutoffs(df, args.round, args.quota, args.category, args.air)

    print("\n=== RESULT ===")
    if out.empty:
        print("No matching colleges found.")
        print("\nTip:")
        print("  • Check that your 'Round' values look like 'Round 1' or 'R1' (or contain 2025_R1).")
        print("  • Quota synonyms like 'All India Quota', '15% AIQ' are auto-mapped to 'AIQ'.")
        print("  • Category OP/UR/GEN → General; BC→OBC; PwD detected via 'PH/PwD' in category or a separate PwD column.")
        print("  • If still stuck, open the new Cutoffs_Normalized sheet (if --write-normalized) and filter there.")
    else:
        cols_show = ["College Name","State","Quota","Round","Category","PwD (Y/N)","ClosingRank"]
        cols_show = [c for c in cols_show if c in out.columns]
        print(out[cols_show].sort_values("ClosingRank").head(30).to_string(index=False))

    if args.write_normalized:
        # Write a clean sheet
        cleaned = df.copy()
        # Build a friendly Category column rebuilt from normalized parts
        cleaned["Category_Normalized"] = cleaned["__BaseCategory"].map({
            "GENERAL":"General", "EWS":"EWS", "OBC":"OBC", "SC":"SC", "ST":"ST"
        })
        cleaned["PwD_Normalized"] = cleaned["__IsPwD"].map({True:"Y", False:"N", None:""})
        # Round compact
        cleaned["Round_Normalized"] = cleaned["__CanonRound"]
        cleaned["Year_Normalized"] = cleaned["__Year"]
        cleaned["NameKey"] = cleaned["__NameKey"]

        with pd.ExcelWriter(args.file, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            cleaned.to_excel(w, sheet_name="Cutoffs_Normalized", index=False)
        print("\nWrote normalized sheet → 'Cutoffs_Normalized' in the same workbook.")

if __name__ == "__main__":
    main()

