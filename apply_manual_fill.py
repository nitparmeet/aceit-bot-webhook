#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
apply_manual_fill.py
--------------------
Apply manual/fuzzy suggestions to fill ClosingRank in your TARGET workbook's Cutoffs sheet,
using source ranks from the patched Table 2 sheet (or any normalized long-form source).

What this script does:
1) Reads SOURCE (e.g., "Cutoffs_Patched_From_Table2") and normalizes keys.
2) Reads TARGET ("Cutoffs" sheet in MCC_Final_with_Cutoffs_2024_2025.xlsx) and normalizes keys.
3) Reads a suggestion CSV (from make_unmatched_suggestions.py) OR can work with blank suggestions.
4) For each suggestion row:
   - Build (College, Quota, Category, PwD, Round) target key
   - Find ClosingRank in SOURCE using a tolerant fallback search:
        a) exact (college, quota, category, pwd, round)
        b) relax quota
        c) relax round
        d) relax both
        e) last resort: any round/any quota max for that college+category+pwd
   - Update TARGET ClosingRank where appropriate.

Usage example:
  python3 apply_manual_fill.py \
    --source-file "Cutoffs MCC 2025 R1_August.xlsx" \
    --source-sheet "Cutoffs_Patched_From_Table2" \
    --target-file "MCC_Final_with_Cutoffs_2024_2025.xlsx" \
    --target-sheet "Cutoffs" \
    --map-csv unmatched_with_suggestions.csv \
    --year 2025 \
    --round R1 \
    --out-sheet "Cutoffs_FuzzyFilled_Manual"

"""

import argparse
import re
from typing import Optional
import pandas as pd
from unidecode import unidecode

# --------------------------
# Helpers
# --------------------------

def main_menu_keyboard():
    return ReplyKeyboardMarkup(
        [
            ["🏥 NEET College Predictor"],
            ["📝 Daily Quiz"],
            ["💬 Clear your NEET Doubts"],
            ["⚙️ Setup your profile"],
        ],
        resize_keyboard=True,
        one_time_keyboard=False
    )




def _t(x) -> str:
    if x is None:
        return ""
    return unidecode(str(x)).strip()

def _name_key(name: str) -> str:
    s = unidecode(str(name or "")).upper()
    s = re.sub(r"\bGOVT\b", "GOVERNMENT", s)
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _canon_category_base(cat: str) -> str:
    t = (cat or "").strip().upper()
    if t in {"UR", "GEN", "GENERAL", "OP", "OPEN"}: return "GENERAL"
    if t in {"EWS"}: return "EWS"
    if t in {"OBC", "BC"}: return "OBC"
    if t in {"SC"}: return "SC"
    if t in {"ST"}: return "ST"
    return t  # last resort

def _canon_pwd(v: str) -> str:
    t = _t(v).upper()
    if t in {"Y", "YES", "TRUE", "1", "PWD", "PH"}:
        return "Y"
    return "N"

def _canon_quota_ui_to_store(q: str) -> str:
    """
    Map UI selections into the *stored* buckets you normalized in Excel.
    - Open Seat / All India → AIQ
    - Deemed/Management/Paid → Management  (if your sheet uses 'Deemed' instead, set to 'Deemed')
    - Central Inst/Univ     → Central
    - ESIC / AFMS / AIIMS / JIPMER → themselves
    """
    t = (q or "").strip().upper()
    if t in {"AIQ", "ALL INDIA", "ALL INDIA QUOTA", "OPEN SEAT QUOTA"}:
        return "AIQ"
    if "DEEMED" in t or "MANAGEMENT" in t or "PAID" in t:
        # If your sheet stores 'Deemed' instead of 'Management', return 'Deemed'
        return "Management"
    if "CENTRAL" in t or t == "CU":
        return "Central"
    if "ESIC" in t: return "ESIC"
    if "AFMS" in t or "ARMED" in t: return "AFMS"
    if "AIIMS" in t: return "AIIMS"
    if "JIPMER" in t: return "JIPMER"
    if "STATE" in t or t in {"SQ", "IP", "IPU"}: return "State"
    return t

def _canon_round_tag(r: str, year: int | None) -> str:
    """
    Accepts 'R1', 'Round 1', '2025_R1' → returns the compact tag used in your sheet.
    If your sheet stores '2025_R1', we still match both.
    """
    v = (r or "").strip().upper()
    # mop/stray pass-through
    if "STRAY" in v: return "STRAY"
    if "MOP" in v: return "MOPUP"
    import re
    m = re.search(r"R(?:OUND)?\s*[-_ ]?\s*(\d+)", v)
    if m:
        n = m.group(1)
        return f"R{n}" if not year else f"{year}_R{n}"
    # already like 2025_R1
    if re.search(r"^\d{4}_R\d+$", v): return v
    if re.search(r"^R\d+$", v): return v
    return v  # last resort

def get_closing_rank_strict(
    df_cutoffs,                    # the loaded Cutoffs DF
    college_name: str,
    quota_ui: str,
    category_ui: str,
    pwd_flag: bool,
    round_ui: str,
    year: int | None = None,
):
    """
    Returns the single ClosingRank (int) for the exact (college, quota, category base, pwd, round, year)
    with Course forced to MBBS. Adds defensive fallbacks and prints a short log line
    when multiple/zero rows are found.
    """
    import pandas as pd

    # Canonicalize filters
    cat_base = _canon_category_base(category_ui)
    quota_c  = _canon_quota_ui_to_store(quota_ui)
    # Try both '2025_R1' and 'R1' forms, because sheets vary
    rt_primary = _canon_round_tag(round_ui, year)
    rt_fallback = _canon_round_tag(round_ui, None)

    # Name-key normalization similar to your loader
    
    def _name_key(s: str) -> str:
    """Same normalization you use elsewhere for reliable joins."""
        if s is None:
            return ""
        import re
        from unidecode import unidecode
        s = unidecode(str(s)).upper()
        s = re.sub(r"\bGOVT\b", "GOVERNMENT", s)
        s = re.sub(r"[^A-Z0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

        
    
    key = _name_key(college_name)

    def _quota_bucket_from_ui(quota_ui: str):
    if quota_ui == "Any":
        return ["AIQ", "Central", "Management", "AIIMS", "JIPMER", "State"]
    return [quota_ui]
    

    def _pick_category_key(category_ui: str) -> str:
    if not category_ui:
        return "General"
    t = category_ui.strip().lower()
    if t in {"gen", "ur", "op", "general"}:
        return "General"
    if t in {"ews"}:
        return "EWS"
    if t in {"obc", "bc"}:
        return "OBC"
    if t in {"sc"}:
        return "SC"
    if t in {"st"}:
        return "ST"
    return category_ui.title()

    def _closing_from_dict(namekey: str, round_key: str, quota_bucket: str, category_key: str):
    """
    Pull closing rank from CUTOFFS_Q[round][quota][namekey][category].
    If quota is 'Any', try buckets in a priority order.
    Returns int or None.
    """
    d_round = (CUTOFFS_Q or {}).get(round_key, {})
    def _get(qb: str):
        v = ((d_round.get(qb) or {}).get(namekey) or {}).get(category_key)
        try:
            return int(v) if v is not None and pd.notna(v) else None
        except Exception:
            return None

    if quota_bucket == "Any":
        for qb in ("AIQ","Central","Management","AIIMS","JIPMER","ESIC","AFMS","State"):
            v = _get(qb)
            if v is not None:
                return v
        return None
    return _get(quota_bucket)

    
    # Ensure numeric ClosingRank
    df = df_cutoffs.copy()
    if "ClosingRank_num" not in df.columns:
        df["ClosingRank_num"] = pd.to_numeric(df["ClosingRank"], errors="coerce")

    # Build mask (strict)
    m = (
        (df.get("__NameKey", df["College Name"].map(_name_key)) == key)
        & (df["Course"].astype(str).str.upper() == "MBBS")
        & (df["Category"].astype(str).map(_canon_category_base) == cat_base)
        & (df["Quota"].astype(str).map(_canon_quota_ui_to_store) == quota_c)
        & (df["ClosingRank_num"].notna())
    )

    # Round/Year handling
    round_col = "Round"
    year_col  = "Year" if "Year" in df.columns else None

    if round_col in df.columns:
        m_round = (df[round_col].astype(str).str.upper().eq(rt_primary))
        if not m_round.any():
            m_round = (df[round_col].astype(str).str.upper().eq(rt_fallback))
        m = m & m_round

    if year and year_col:
        m = m & ((pd.to_numeric(df[year_col], errors="coerce") == year) | df[year_col].isna())

    view = df.loc[m, ["College Name","Quota","Category","PwD (Y/N)","Round","Year","ClosingRank_num"]].copy()

    # PwD strictness: treat Y/N column strictly only if user explicitly asked PwD
    if "PwD (Y/N)" in view.columns:
        if pwd_flag:
            view = view[view["PwD (Y/N)"].astype(str).str.upper().isin(["Y","YES","TRUE","1"])]
        else:
            view = view[~view["PwD (Y/N)"].astype(str).str.upper().isin(["Y","YES","TRUE","1"])]

    if view.empty:
        # Debug line to catch mismatches in logs
        print(f"[cutoff-miss] name={college_name!r} key={key} quota={quota_c} cat={cat_base} pwd={pwd_flag} round={rt_primary}/{rt_fallback} year={year}")
        return None

    # If multiple rows (duplicates across sources), pick the **max rank** as closing rank for eligibility
    # (worst rank admitted), but this is also what you want to display in your lists.
    close = int(view["ClosingRank_num"].max())

    # Debug a short proof
    top_row = view.sort_values("ClosingRank_num", ascending=False).iloc[0].to_dict()
    print(f"[cutoff-hit] {college_name} → {close}  ({top_row})")

    return close








# --------------------------
# Read & normalize source / target
# --------------------------

def read_source(path: str, sheet: str, year: int, round_tag: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    # Expected columns (from Cutoffs_Patched_From_Table2): College Name, QuotaCanon, CategoryBase, IsPwD, Year, RoundTag, ClosingRank
    # But we’ll re-derive canon columns to be safe.
    # College
    if "College Name" not in df.columns:
        raise ValueError(f"'College Name' missing in source sheet '{sheet}'")

    # Quota
    if "QuotaCanon" not in df.columns:
        qcol = None
        for c in df.columns:
            if "quota" in _t(c).lower():
                qcol = c
                break
        if qcol:
            df["QuotaCanon"] = df[qcol].map(_canon_quota_from_value)
        else:
            df["QuotaCanon"] = None

    # Category
    if "CategoryBase" not in df.columns:
        ccol = None
        for c in df.columns:
            lc = _t(c).lower()
            if lc in {"category", "cat"}:
                ccol = c
                break
        if ccol:
            df["CategoryBase"] = df[ccol].map(_canon_cat_base)
        else:
            df["CategoryBase"] = "GENERAL"

    # PwD
    if "IsPwD" not in df.columns:
        pcol = None
        for c in df.columns:
            lc = _t(c).lower()
            if "pwd" in lc or lc in {"ph"}:
                pcol = c
                break
        if pcol:
            df["IsPwD"] = df[pcol].map(_canon_pwd)
        else:
            # also detect PWD embedded in category
            tmp = df.get("Category", "")
            df["IsPwD"] = tmp.astype(str).str.contains(r"\b(PWD|PH)\b", case=False, regex=True).map({True: "Y", False: "N"})

    # Year
    if "Year" not in df.columns:
        ycol = None
        for c in df.columns:
            if "year" in _t(c).lower():
                ycol = c
                break
        if ycol:
            df["Year"] = pd.to_numeric(df[ycol], errors="coerce").astype("Int64")
        else:
            df["Year"] = pd.Series([year] * len(df), dtype="Int64")

    # RoundTag
    if "RoundTag" not in df.columns:
        rcol = None
        for c in df.columns:
            if "round" in _t(c).lower():
                rcol = c
                break
        if rcol:
            df["RoundTag"] = df[rcol].map(_canon_round_short)
        else:
            df["RoundTag"] = _canon_round_short(round_tag)

    # ClosingRank
    if "ClosingRank" not in df.columns:
        # common alternatives
        cand = None
        for c in df.columns:
            lc = _t(c).lower().replace(" ", "")
            if lc in {"closingrank", "closing rank", "rank"}:
                cand = c
                break
        if cand is None:
            raise ValueError("Source has no ClosingRank column.")
        df["ClosingRank"] = pd.to_numeric(df[cand], errors="coerce")

    # NameKey
    df["NameKey"] = df["College Name"].map(_name_key)

    # Keep only needed cols
    keep = ["College Name", "NameKey", "QuotaCanon", "CategoryBase", "IsPwD", "Year", "RoundTag", "ClosingRank"]
    df = df[keep].copy()

    # Keep a copy unfiltered (for tolerant lookup across rounds/quotas)
    df_full = df.copy()

    # Also return filtered-to-year for faster primary lookup
    df_year = df[df["Year"].astype("Int64") == int(year)].copy()

    # Return both
    df_year.reset_index(drop=True, inplace=True)
    df_full.reset_index(drop=True, inplace=True)
    return df_year, df_full

def _norm(s: str | None) -> str:
    return (str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")) if s is not None else ""

def _norm_round(s: str | None) -> str:
    # Accept things like 2025_R1, r1_2025, round1_2025 etc.
    s = _norm(s)
    if not s:
        return ""
    # pull year and r#
    year = ""
    r = ""
    for tok in ("2023","2024","2025","2026","2027"):
        if tok in s:
            year = tok
            break
    for i in ("r1","round1","r01"):
        if i in s:
            r = "R1"
            break
    for i in ("r2","round2","r02"):
        if i in s:
            r = "R2"
            break
    for i in ("r3","round3","r03"):
        if i in s:
            r = "R3"
            break
    for i in ("r4","round4","r04"):
        if i in s:
            r = "R4"
            break
    if year and r:
        return f"{year}_{r}"
    # already in a nice form?
    if "r" in s and any(y in s for y in ("2023","2024","2025","2026","2027")):
        # try to keep as-is but uppercase R
        s = s.replace("r","R")
    return s



def _close_rank_tolerant(
    nk: str,
    *,
    quota: str,
    cat: str,
    cutoff_lookup: dict,
    per_round: dict | None = None,  # optional nested structure if you still keep it
) -> int | None:
    """
    Robust closing-rank fetch:
    - tries exact (code, quota, cat)
    - falls back to General, then to AIQ
    - if nested per-round dict exists, also tries that
    Returns int or None.
    """
    if not nk:
        return None

    # 1) flat map – exact
    v = cutoff_lookup.get((nk, quota, cat))
    if isinstance(v, int):
        return v

    # 2) flat map – same quota, General
    if cat != "General":
        v = cutoff_lookup.get((nk, quota, "General"))
        if isinstance(v, int):
            return v

    # 3) flat map – AIQ with our cat
    if quota != "AIQ":
        v = cutoff_lookup.get((nk, "AIQ", cat))
        if isinstance(v, int):
            return v

    # 4) flat map – AIQ + General
    v = cutoff_lookup.get((nk, "AIQ", "General"))
    if isinstance(v, int):
        return v

    # 5) try nested per-round structure if present
    if per_round:
        # per_round is expected like {quota: {code: {cat: rank}}}
        qmap = per_round.get(quota) or per_round.get("AIQ") or {}
        rec = qmap.get(nk) or {}
        # reuse your existing category resolver
        val = _get_close_rank_from_rec(rec, cat)
        if isinstance(val, int):
            return val

    return None

def format_flashcards(cards: list[tuple[str, str]]) -> str:
    """
    Takes a list of (question, answer) and returns formatted flashcards text.
    """
    blocks = []
    for i, (q, a) in enumerate(cards, start=1):
        block = (
            f"<b>Q&A {i}</b>\n"
            f"❓ <i>{q.strip()}</i>\n"
            f"💡 {a.strip()}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)



def read_target(path: str, sheet: str, year: int, round_tag: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)

    # Required columns in target
    for req in ["College Name", "Quota", "Category", "PwD (Y/N)", "Round"]:
        if req not in df.columns:
            raise ValueError(f"Target sheet '{sheet}' missing required column: {req}")

    # Normalize keys
    df["NameKey"] = df["College Name"].map(_name_key)
    df["QuotaCanon"] = df["Quota"].map(_canon_quota_from_value)
    df["CategoryBase"] = df["Category"].map(_canon_cat_base)
    df["IsPwD"] = df["PwD (Y/N)"].map(_canon_pwd)
    df["RoundTag"] = df["Round"].map(_canon_round_short)

    # Ensure ClosingRank exists
    if "ClosingRank" not in df.columns:
        df["ClosingRank"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # Filter by year/round if a Year column exists; otherwise keep all and rely on RoundTag only
    if "Year" in df.columns:
        df = df[df["Year"].astype(str).str.contains(str(year), na=False)]

    df.reset_index(drop=True, inplace=True)
    return df

# --------------------------
# Tolerant lookup
# --------------------------

def lookup_closing_rank_tolerant(src_df_full: pd.DataFrame, name: str, quota: Optional[str],
                                 cat: str, pwd: str, rnd: str) -> Optional[int]:
    """
    Progressive fallback search for ClosingRank in the SOURCE (unfiltered).
    Priority:
      1) exact match on (college, quota, category, pwd, round)
      2) relax quota (any quota)
      3) relax round (any round, keep quota if given)
      4) relax both (any round, any quota)
      5) last resort: any round/any quota for that college/category/pwd (max rank)
    Returns int or None.
    """
    nk = _name_key(name)
    catU = _canon_cat_base(cat)
    pwdY = _canon_pwd(pwd)
    rshort = _canon_round_short(rnd)

    def _pick(view: pd.DataFrame) -> Optional[int]:
        if view.empty:
            return None
        s = pd.to_numeric(view["ClosingRank"], errors="coerce").dropna()
        if s.empty:
            return None
        return int(s.max())

    # 1) full key
    view = src_df_full[
        (src_df_full["NameKey"] == nk) &
        (src_df_full["CategoryBase"] == catU) &
        (src_df_full["IsPwD"] == pwdY) &
        (src_df_full["RoundTag"].str.upper() == rshort.upper()) &
        ((src_df_full["QuotaCanon"] == quota) if quota else True)
    ]
    v = _pick(view)
    if v is not None:
        return v

    # 2) relax quota
    view = src_df_full[
        (src_df_full["NameKey"] == nk) &
        (src_df_full["CategoryBase"] == catU) &
        (src_df_full["IsPwD"] == pwdY) &
        (src_df_full["RoundTag"].str.upper() == rshort.upper())
    ]
    v = _pick(view)
    if v is not None:
        return v

    # 3) relax round
    view = src_df_full[
        (src_df_full["NameKey"] == nk) &
        (src_df_full["CategoryBase"] == catU) &
        (src_df_full["IsPwD"] == pwdY) &
        ((src_df_full["QuotaCanon"] == quota) if quota else True)
    ]
    v = _pick(view)
    if v is not None:
        return v

    # 4) relax both (any round, any quota)
    view = src_df_full[
        (src_df_full["NameKey"] == nk) &
        (src_df_full["CategoryBase"] == catU) &
        (src_df_full["IsPwD"] == pwdY)
    ]
    v = _pick(view)
    if v is not None:
        return v

    # 5) last resort: same as (4) – already max
    return None

     # ---- sort & limit
    df_sorted = df.sort_values("__Close", ascending=True)
    top15 = df_sorted.head(15)

    # pick columns present
    col_name   = "College Name" if "College Name" in df_sorted.columns else df_sorted.columns[0]
    col_state  = "State" if "State" in df_sorted.columns else None
    col_site   = "Website" if "Website" in df_sorted.columns else None
    col_nirf   = "NIRF" if "NIRF" in df_sorted.columns else None
    col_fee    = "Fee" if "Fee" in df_sorted.columns else None
    col_bond   = "Has_Bond" if "Has_Bond" in df_sorted.columns else None
    col_pg     = "Has_PG" if "Has_PG" in df_sorted.columns else None
    col_same   = "SameState" if "SameState" in df_sorted.columns else None
    col_quota  = "QuotaCanon" if "QuotaCanon" in df_sorted.columns else None
    col_cat    = "CategoryBase" if "CategoryBase" in df_sorted.columns else None

    def _safe_int(x):
    try:
        if pd.isna(x): return None
        return int(float(x))
    except Exception:
        return None

    def _safe_float(x):
    try:
        if pd.isna(x): return None
        return float(x)
    except Exception:
        return None


    out = []
    for _, r in df_sorted.iterrows():
        rec = {
            "College Name": (str(r.get(col_name)) if col_name else None),
            "State":        (str(r.get(col_state)) if col_state else None),
            "ClosingRank":  _safe_rank(r.get("__Close")),
            "Category":     (str(r.get(col_cat)) if col_cat else cat_key),
            "Quota":        (str(r.get(col_quota)) if col_quota else bucket),
            "Score":        _safe_float(r.get("__Score")),
            "Website":      (str(r.get(col_site)) if col_site else None),
            "NIRF":         _safe_rank(r.get(col_nirf)) if col_nirf else None,
            "Fee":          _safe_rank(r.get(col_fee)) if col_fee else None,
            "Has_Bond":     bool(r.get(col_bond)) if col_bond else False,
            "Has_PG":       bool(r.get(col_pg)) if col_pg else False,
            "SameState":    bool(r.get(col_same)) if col_same else False,
        }
        # tidy strings
        if rec["College Name"]:
            rec["College Name"] = rec["College Name"].strip()
        if rec["State"]:
            rec["State"] = rec["State"].strip()
        out.append(rec)

    logger.info("shortlist: returning %d rows", len(out))
    logger.info("round=%s bucket=%s cat=%s | cutoff-keys=%d",
            round_ui, bucket, cat_key, len(cut_map))

    logger.info("sample cutoff keys: %s", list(cut_map.keys())[:5])
    return out

EXCEL_PATH = "MCC_Final_with_Cutoffs_2024_2025.xlsx" 
    CUTOFF_LOOKUP = load_cutoff_lookup_from_excel(
    path=EXCEL_PATH,
    sheet="Cutoffs",
    year=2025,
    round_tag="2025_R1",
    course_contains="MBBS",
)

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-file", required=True, help="Workbook with source ranks (e.g., Cutoffs MCC 2025 R1_August.xlsx)")
    ap.add_argument("--source-sheet", default="Cutoffs_Patched_From_Table2", help="Sheet in source file (default: Cutoffs_Patched_From_Table2)")
    ap.add_argument("--target-file", required=True, help="Workbook to update (e.g., MCC_Final_with_Cutoffs_2024_2025.xlsx)")
    ap.add_argument("--target-sheet", default="Cutoffs", help="Target sheet name in target file (default: Cutoffs)")
    ap.add_argument("--map-csv", required=True, help="CSV produced by make_unmatched_suggestions.py")
    ap.add_argument("--year", type=int, required=True, help="Year, e.g. 2025")
    ap.add_argument("--round", required=True, help="Round tag, e.g. R1")
    ap.add_argument("--out-sheet", default="Cutoffs_FuzzyFilled_Manual", help="Output sheet name in target")
    args = ap.parse_args()

    # Load source (both year-filtered and full for fallbacks)
    src_year, src_full = read_source(args.source_file, args.source_sheet, args.year, args.round)

    # Load target
    tgt = read_target(args.target_file, args.target_sheet, args.year, args.round)

    # Load mapping CSV
    mapping = pd.read_csv(args.map_csv)

    # Which columns might the CSV contain (from the suggestion tool)
    # We’ll read the “Source_*” and “Suggested_*” sets if present, else try direct target row values.
    def pick(colname: str) -> Optional[str]:
        return colname if colname in mapping.columns else None

    source_cols = {
        "name": pick("Source_College"),
        "quota": pick("Source_Quota"),
        "cat": pick("Source_Category"),
        "pwd": pick("Source_PwD"),
        "round": pick("Source_RoundTag"),
    }
    sug_cols = {
        "name": pick("Suggested_College"),
        "quota": pick("Suggested_Quota"),
        "cat": pick("Suggested_Category"),
        "pwd": pick("Suggested_PwD"),
        "round": pick("Suggested_RoundTag"),
    }
    tgt_cols = {
        "row_index": pick("TargetRowIndex"),
        "name": pick("Target_College"),
        "quota": pick("Target_Quota"),
        "cat": pick("Target_Category"),
        "pwd": pick("Target_PwD"),
        "round": pick("Target_RoundTag"),
    }

    updated = 0
    reasons = {"no_source_key": 0, "no_target_pick": 0, "no_target_rows": 0}

    # We’ll update a copy so we can write a new sheet
    out_df = tgt.copy()

    for _, row in mapping.iterrows():
        # 1) choose target row to update
        tgt_idx = None
        if tgt_cols["row_index"] and pd.notna(row[tgt_cols["row_index"]]):
            try:
                tgt_idx = int(row[tgt_cols["row_index"]])
            except Exception:
                tgt_idx = None

        # If row index was not provided, try to locate target row by target columns (fallback)
        if tgt_idx is None:
            # Try a keyed filter in the target DF
            # Prefer Suggested_* for matching if present; else fall back to target columns in CSV; else skip.
            t_name = row[sug_cols["name"]] if sug_cols["name"] else (row[tgt_cols["name"]] if tgt_cols["name"] else None)
            t_quota = row[sug_cols["quota"]] if sug_cols["quota"] else (row[tgt_cols["quota"]] if tgt_cols["quota"] else None)
            t_cat = row[sug_cols["cat"]] if sug_cols["cat"] else (row[tgt_cols["cat"]] if tgt_cols["cat"] else None)
            t_pwd = row[sug_cols["pwd"]] if sug_cols["pwd"] else (row[tgt_cols["pwd"]] if tgt_cols["pwd"] else None)
            t_round = row[sug_cols["round"]] if sug_cols["round"] else (row[tgt_cols["round"]] if tgt_cols["round"] else None)

            if pd.isna(t_name):
                reasons["no_target_pick"] += 1
                continue

            # Filter target rows
            view = out_df[
                (out_df["NameKey"] == _name_key(t_name)) &
                ((out_df["QuotaCanon"] == _canon_quota_from_value(t_quota)) if pd.notna(t_quota) else True) &
                ((out_df["CategoryBase"] == _canon_cat_base(t_cat)) if pd.notna(t_cat) else True) &
                ((out_df["IsPwD"] == _canon_pwd(t_pwd)) if pd.notna(t_pwd) else True) &
                ((out_df["RoundTag"] == _canon_round_short(t_round)) if pd.notna(t_round) else True)
            ]
            if view.empty:
                reasons["no_target_rows"] += 1
                continue
            tgt_idx = int(view.index[0])

        # 2) get source key (prefer Source_* if present; else fall back to the target row values)
        s_name = row[source_cols["name"]] if source_cols["name"] and pd.notna(row[source_cols["name"]]) else out_df.at[tgt_idx, "College Name"]
        s_quota_raw = row[source_cols["quota"]] if source_cols["quota"] and pd.notna(row[source_cols["quota"]]) else out_df.at[tgt_idx, "Quota"]
        s_cat_raw = row[source_cols["cat"]] if source_cols["cat"] and pd.notna(row[source_cols["cat"]]) else out_df.at[tgt_idx, "Category"]
        s_pwd_raw = row[source_cols["pwd"]] if source_cols["pwd"] and pd.notna(row[source_cols["pwd"]]) else out_df.at[tgt_idx, "PwD (Y/N)"]
        s_round_raw = row[source_cols["round"]] if source_cols["round"] and pd.notna(row[source_cols["round"]]) else out_df.at[tgt_idx, "Round"]

        s_quota = _canon_quota_from_value(s_quota_raw)
        s_cat = _canon_cat_base(s_cat_raw)
        s_pwd = _canon_pwd(s_pwd_raw)
        s_round = _canon_round_short(s_round_raw)

        # 3) tolerant lookup in source (use FULL src for better chances)
        close_val = lookup_closing_rank_tolerant(
            src_df_full=src_full,
            name=s_name,
            quota=s_quota,
            cat=s_cat,
            pwd=s_pwd,
                rnd=s_round
        )
        if close_val is None:
            reasons["no_source_key"] += 1
            continue

        # 4) write into target row
        out_df.at[tgt_idx, "ClosingRank"] = int(close_val)
        updated += 1

    # Write the output sheet back into target workbook
    with pd.ExcelWriter(args.target_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
        out_df.to_excel(xw, sheet_name=args.out_sheet, index=False)

    print(f"[done] Wrote '{args.out_sheet}' into {args.target_file}  |  rows updated: {updated}")
    if updated == 0:
        print("  └─ No rows updated. Common reasons:")
        print("     • Source key didn’t exist for that (College, Quota, Category, PwD, Round)")
        print("     • No target row matched (Name/Quota/Category/PwD/Round)")
        print("     • Suggestion columns empty and no direct name match found")
    print(f"[why not updated breakdown] {reasons}")


if __name__ == "__main__":
    main()
