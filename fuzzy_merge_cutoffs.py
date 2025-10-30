import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import fuzz, process
from unidecode import unidecode
import re

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first df column whose normalized name matches or contains any of the candidate tokens.
    """
    def norm(s: str) -> str:
        s2 = unidecode(str(s)).lower()
        s2 = re.sub(r"[^a-z0-9 ]+", " ", s2)
        s2 = re.sub(r"\s+", " ", s2).strip()
        return s2

    cols_norm = {c: norm(c) for c in df.columns}
    cands_norm = [norm(c) for c in candidates]

    for c, cn in cols_norm.items():
        for cand in cands_norm:
            if cn == cand or cand in cn:
                return c
    return None

# ----------------------------
# Helpers
# ----------------------------

def _name_key(name: str) -> str:
    if name is None:
        return ""
    s = unidecode(str(name)).upper()

    # normalize abbreviations
    s = re.sub(r"\bGOVT\b", "GOVERNMENT", s)
    repls = {
        "INST.OF": "INSTITUTE OF",
        "INST OF": "INSTITUTE OF",
        "INST. OF": "INSTITUTE OF",
        "MED.SCIENCES": "MEDICAL SCIENCES",
        "MED. SCIENCES": "MEDICAL SCIENCES",
        "MED SCIENCES": "MEDICAL SCIENCES",
        "COLL.": "COLLEGE",
        "COLL ": "COLLEGE ",
        "BHU": "BANARAS HINDU UNIVERSITY",
        "&": " AND ",
    }
    for k, v in repls.items():
        s = s.replace(k, v)

    # remove campus suffixes
    s = re.sub(r"\bCAMPUS\b.*$", "", s)

    # strip trailing numeric codes
    s = re.sub(r"\(\s*\d{4,6}\s*\)$", "", s)

    # clean
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _canon_category_base(x: str):
    t = (unidecode(str(x)).upper().strip() if x is not None else "")
    t = re.sub(r"[^A-Z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t)

    if not t:
        return None

    t = re.sub(r"\b(PWD|PH)\b", "", t).strip()

    if t in {"UR", "GEN", "GENERAL", "OP", "OC", "OPEN"}:
        return "General"
    if t in {"EWS"}:
        return "EWS"
    if t in {"OBC", "BC"}:
        return "OBC"
    if t in {"SC"}:
        return "SC"
    if t in {"ST"}:
        return "ST"

    toks = t.split()
    if any(tok in {"UR", "GEN", "GENERAL", "OP", "OC", "OPEN"} for tok in toks):
        return "General"
    if "EWS" in toks:
        return "EWS"
    if any(tok in {"OBC", "BC"} for tok in toks):
        return "OBC"
    if "SC" in toks:
        return "SC"
    if "ST" in toks:
        return "ST"

    return "General"


def _canon_quota_from_value(val):
    if val is None:
        return None
    t = unidecode(str(val)).upper().strip()

    if t in {"AIQ", "ALLINDIAQUOTA", "ALLINDIA", "15 AIQ", "15 PERCENT AIQ"}:
        return "AIQ"
    if ("ALL" in t and "INDIA" in t) or "AIQ" in t:
        return "AIQ"
    if "CENTRAL" in t or "CU" == t or "CENTRAL INSTITUTION" in t or "CENTRAL UNIVERSITY" in t:
        return "Central"
    if "DEEMED" in t or "MANAGEMENT" in t or "PAID" in t:
        return "Management"
    if "ESIC" in t or "ESI" in t:
        return "ESIC"
    if "AFMS" in t or "ARMED" in t or "FORCES" in t:
        return "AFMS"
    if "STATE" in t or t in {"SQ", "IP", "IPU", "STATEQUOTA", "DOMICILE", "HOMESTATE"}:
        return "State"
    if "AIIMS" in t:
        return "AIIMS"
    if "JIPMER" in t:
        return "JIPMER"

    return None


def _bucket(quota, cat, pwd, year, roundtag):
    return f"{quota or ''}|{cat or ''}|{pwd or ''}|{year or ''}|{roundtag or ''}"


# ----------------------------
# Reading
# ----------------------------

def read_source(path: str, sheet: str, year: int, roundtag: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)

    # Flexible header detection
    col_name   = find_col(df, ["college name", "college", "institute", "inst name", "institute name"])
    col_quota  = find_col(df, ["quotacanon", "quota", "all india quota", "aiq", "seat quota"])
    col_cat    = find_col(df, ["categorybase", "category", "cat"])
    col_pwd    = find_col(df, ["pwd", "pwd (y/n)", "ph", "is ph", "is_pwd"])
    col_year   = find_col(df, ["year", "session"])
    col_round  = find_col(df, ["roundtag", "round"])
    col_rank   = find_col(df, ["closingrank", "closing rank", "rank", "closing rank (air)"])

    if not col_name:
        raise ValueError("Source sheet: could not find a 'College Name' column.")

    # Normalize core fields
    df["College Name"] = df[col_name]
    df["NameKey"] = df["College Name"].map(_name_key)

    # Quota
    if col_quota:
        df["QuotaCanon"] = df[col_quota].map(_canon_quota_from_value)
    else:
        # default AIQ if absent
        df["QuotaCanon"] = "AIQ"

    # Category
    if col_cat:
        df["CategoryBase"] = df[col_cat].map(_canon_category_base)
    else:
        df["CategoryBase"] = "General"

    # PwD
    if col_pwd:
        df["IsPwD"] = df[col_pwd].astype(str).str.upper().map(lambda x: "Y" if x in {"Y","YES","1","TRUE"} else "N")
    else:
        df["IsPwD"] = "N"

    # Year / Round
    if col_year:
        df["Year"] = pd.to_numeric(df[col_year], errors="coerce").fillna(year).astype(int)
    else:
        df["Year"] = int(year)

    df["RoundTag"] = (df[col_round] if col_round else roundtag)
    df["RoundTag"] = df["RoundTag"].astype(str).str.upper().str.strip()

    # Closing rank (optional in source)
    if col_rank:
        df["ClosingRank"] = pd.to_numeric(df[col_rank], errors="coerce")
    else:
        # keep column for merge; may be missing for some rows
        df["ClosingRank"] = pd.NA

    # Keep only the columns we use downstream
    keep_cols = ["College Name", "NameKey", "QuotaCanon", "CategoryBase", "IsPwD", "Year", "RoundTag", "ClosingRank"]
    return df[keep_cols].copy()


def read_target(path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)

    col_name   = find_col(df, ["college name", "college", "institute", "inst name", "institute name"])
    col_quota  = find_col(df, ["quotacanon", "quota", "all india quota", "aiq", "seat quota"])
    col_cat    = find_col(df, ["categorybase", "category", "cat"])
    col_pwd    = find_col(df, ["pwd (y/n)", "pwd", "ph", "is ph", "is_pwd"])
    col_year   = find_col(df, ["year", "session"])
    col_round  = find_col(df, ["roundtag", "round"])
    col_rank   = find_col(df, ["closingrank", "closing rank", "rank", "closing rank (air)"])

    if not col_name:
        raise ValueError("Target sheet: could not find a 'College Name' column.")

    # Normalize
    df["College Name"] = df[col_name]
    df["NameKey"] = df["College Name"].map(_name_key)

    if col_quota:
        # If already canonical, _canon_quota_from_value will leave it fine
        df["QuotaCanon"] = df[col_quota].map(_canon_quota_from_value)
    else:
        df["QuotaCanon"] = "AIQ"

    if col_cat:
        df["CategoryBase"] = df[col_cat].map(_canon_category_base)
    else:
        df["CategoryBase"] = "General"

    if col_pwd:
        df["IsPwD"] = df[col_pwd].astype(str).str.upper().map(lambda x: "Y" if x in {"Y","YES","1","TRUE"} else "N")
    else:
        df["IsPwD"] = "N"

    if col_year:
        df["Year"] = pd.to_numeric(df[col_year], errors="coerce")
    else:
        df["Year"] = pd.NA

    if col_round:
        df["RoundTag"] = df[col_round].astype(str).str.upper().str.strip()
    else:
        df["RoundTag"] = pd.NA

    if col_rank and col_rank != "ClosingRank":
        df = df.rename(columns={col_rank: "ClosingRank"})
    if "ClosingRank" not in df.columns:
        df["ClosingRank"] = pd.NA

    # Preserve any extra columns for writing back, but ensure our merge keys exist
    return df.copy()


# ----------------------------
# Merge Logic
# ----------------------------

def merge_cutoffs(src, tgt, fuzzy_threshold=90):
    out = tgt.copy()

    def bucket_full(r):
        return _bucket(r["QuotaCanon"], r["CategoryBase"], r["IsPwD"], r["Year"], r["RoundTag"])

    def bucket_no_quota(r):
        return _bucket(None, r["CategoryBase"], r["IsPwD"], r["Year"], r["RoundTag"])

    def bucket_no_pwd(r):
        return _bucket(r["QuotaCanon"], r["CategoryBase"], None, r["Year"], r["RoundTag"])

    src_by_name = {}
    for _, r in src.iterrows():
        nk = r["NameKey"]
        src_by_name.setdefault(nk, []).append(r)
    src_keys = list(src_by_name.keys())

    def find_rank(row, mode):
        want_quota, want_cat, want_pwd = row["QuotaCanon"], row["CategoryBase"], row["IsPwD"]
        want_year, want_round = row.get("Year"), row.get("RoundTag")

        if mode == "full":
            want_bucket = _bucket(want_quota, want_cat, want_pwd, want_year, want_round)
        elif mode == "no_quota":
            want_bucket = _bucket(None, want_cat, want_pwd, want_year, want_round)
        else:  # no_pwd
            want_bucket = _bucket(want_quota, want_cat, None, want_year, want_round)

        key_guess = row["NameKey"]

        # Direct match
        if key_guess in src_by_name:
            for cand in src_by_name[key_guess]:
                b = (bucket_full(cand) if mode == "full"
                     else bucket_no_quota(cand) if mode == "no_quota"
                     else bucket_no_pwd(cand))
                if b == want_bucket:
                    return cand.get("ClosingRank")

        # Fuzzy fallback
        match = process.extractOne(key_guess, src_keys, scorer=fuzz.token_sort_ratio)
        if not match:
            return None
        match_name, score = match[0], match[1]
        need = fuzzy_threshold if mode == "full" else max(80, fuzzy_threshold - 5)
        if score < need:
            return None

        for cand in src_by_name[match_name]:
            b = (bucket_full(cand) if mode == "full"
                 else bucket_no_quota(cand) if mode == "no_quota"
                 else bucket_no_pwd(cand))
            if b == want_bucket:
                return cand.get("ClosingRank")
        return None

    unmatched = []

    for i, row in out.iterrows():
        rank = find_rank(row, "full")
        if rank is None and (row.get("QuotaCanon") is None or str(row.get("QuotaCanon") or "") == ""):
            rank = find_rank(row, "no_quota")
        if rank is None:
            rank = find_rank(row, "no_pwd")

        if rank is not None and not pd.isna(rank):
            try:
                out.at[i, "ClosingRank"] = int(rank)
            except Exception:
                out.at[i, "ClosingRank"] = pd.NA
        else:
            unmatched.append({
                "College Name": row.get("College Name"),
                "NameKey": row.get("NameKey"),
                "QuotaCanon": row.get("QuotaCanon"),
                "CategoryBase": row.get("CategoryBase"),
                "IsPwD": row.get("IsPwD"),
                "Year": row.get("Year"),
                "RoundTag": row.get("RoundTag"),
            })

    return out, pd.DataFrame(unmatched)


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-file", required=True)
    ap.add_argument("--source-sheet", default="Cutoffs_Patched_From_Table2")
    ap.add_argument("--target-file", required=True)
    ap.add_argument("--target-sheet", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", required=True)
    ap.add_argument("--out-sheet", default="Cutoffs_FuzzyFilled")
    ap.add_argument("--fuzzy-threshold", type=int, default=90)
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"[info] Reading source → {args.source_file}")
    src = read_source(args.source_file, args.source_sheet, args.year, args.round)
    print(f"[info] Source rows (grouped): {len(src)}")

    print(f"[info] Reading target → {args.target_file} / sheet '{args.target_sheet}'")
    tgt = read_target(args.target_file, args.target_sheet)
    print(f"[info] Target rows: {len(tgt)}")

    merged, unmatched = merge_cutoffs(src, tgt, fuzzy_threshold=args.fuzzy_threshold)

    out_path = Path(args.target_file)
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
        merged.to_excel(xw, sheet_name=args.out_sheet, index=False)

    unmatched_file = f"unmatched_for_{args.out_sheet}.csv"
    unmatched.to_csv(unmatched_file, index=False)

    print(f"[done] Wrote merged sheet → '{args.out_sheet}' in {out_path.name}")
    print(f"[done] Unmatched rows: {len(unmatched)} → {unmatched_file}")


if __name__ == "__main__":
    main()
