#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import secrets





from telegram.ext import Application




from pathlib import Path   # <-- add this line




import contextlib


import httpx
from telegram.error import BadRequest


import json
import logging
import html
import asyncio
import math
import random
import time
import base64
from openai import OpenAI
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram import ReplyKeyboardMarkup
from telegram.constants import ChatAction
from typing import Dict, Any, List, Optional, Tuple, Iterable
from telegram import Bot
import pandas as pd
from dotenv import load_dotenv
from unidecode import unidecode
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler, CallbackQueryHandler, ConversationHandler






log = logging.getLogger("aceit-bot")
TOKEN = os.getenv("TELEGRAM_TOKEN", "")
QUIZ_SESSIONS: Dict[int, Dict[str, Any]] = {}
from dataclasses import dataclass
QUIZ_POOL: List[Dict[str, Any]] = []  
QUIZ_INDEX: Dict[str, Dict[str, Any]] = {} 
QUIZ_FILE_PATH = Path(__file__).parent / "quiz.json"

from dataclasses import dataclass
from pydantic import BaseModel, ValidationError, Field
CUTOFF_LOOKUP: dict = {}
COLLEGES: list = []
COLLEGE_META_INDEX: dict = {}

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None  # type: ignore
    logging.getLogger(__name__).exception("OpenAI SDK import failed: %s", e)

_openai_client: Optional["OpenAI"] = None



_client_singleton = None


def _new_token(n=8) -> str:
    # 8 hex chars, upper → short but unique per session
    return secrets.token_hex(n//2).upper()
    
def _stack_keyboards(*markups: InlineKeyboardMarkup | None) -> InlineKeyboardMarkup | None:
    rows = []
    for m in markups:
        if isinstance(m, InlineKeyboardMarkup):
            rows.extend(m.inline_keyboard or [])
    return InlineKeyboardMarkup(rows) if rows else None

def _safe_upper(x):
    return (str(x or "")).strip().upper()

_QUOTA_ALIASES = {
    "AIQ": {"AIQ", "Central", "All India Quota"},
}
_CATEGORY_ALIASES = {
    "OP": {"OP", "Open", "UR", "General"},
}

NA_STRINGS = {"", "—", "-", "na", "n/a", "nan", "none", "null"}

def _safe_str(v, default: str = "") -> str:
    try:
        if v is None: return default
        if isinstance(v, float) and math.isnan(v): return default
        s = str(v).strip()
        return default if s.lower() in NA_STRINGS else s
    except Exception:
        return default

def _norm_token(x: str | None) -> str | None:
    if x is None: return None
    return str(x).strip()

def _variants_for_quota(q: str | None) -> set[str]:
    q = _norm_token(q)
    if not q: return {None}
    for k, s in _QUOTA_ALIASES.items():
        if q in s or q == k:
            return set(s) | {k}
    return {q}

def _variants_for_cat(c: str | None) -> set[str]:
    c = _norm_token(c)
    if not c: return {None}
    for k, s in _CATEGORY_ALIASES.items():
        if c in s or c == k:
            return set(s) | {k}
    return {c}

_CODE_RE = re.compile(r"^[Cc]?\d{2,5}$")  # e.g. C0065, 0065, 65

def _norm_code_like(s: str) -> tuple[str, str]:
    """Return (original upper, only-digits) for matching code-ish keys."""
    s1 = str(s).strip().upper()
    s2 = re.sub(r"\D+", "", s1)
    return s1, s2

def _value_to_closing(v):
    """Extract a closing-rank-like value from numbers/strings/dicts."""
    if v is None: return None
    if isinstance(v, (int, float)): return int(v)
    if isinstance(v, str):
        v = v.strip()
        if not v: return None
        try: return int(float(v))
        except: return v
    if isinstance(v, dict):
        for k in ("ClosingRank","closing_rank","closing","rank","CR"):
            if k in v and v[k] not in (None,"","—","-","NA"):
                return _value_to_closing(v[k])
    return None
def _closing_rank_for_identifiers(identifiers: list[str],
                                  round_code: str | None,
                                  quota: str | None,
                                  category: str | None,
                                  df_lookup,  # may be None or empty
                                  lookup_dict: dict | None):
    """Try many shapes in CUTOFFS_DF / CUTOFF_LOOKUP for these ids."""
    ids = [i for i in (identifiers or []) if i]
    if not ids:
        return None

    # Prepare normalized id variants (raw + digits-only for code-like)
    id_variants: set[str] = set()
    for i in ids:
        i1 = str(i).strip()
        id_variants.add(i1)
        up, digits = _norm_code_like(i1)
        if digits: id_variants.add(digits)
        id_variants.add(up)

    r_variants = {round_code} if round_code else {None}
    q_variants = _variants_for_quota(quota)
    c_variants = _variants_for_cat(category)

    # 1) Try DataFrame (if present)
    try:
        if df_lookup is not None and hasattr(df_lookup, "empty") and not df_lookup.empty:
            df = df_lookup.copy()
            # try matching across several possible identifier columns
            id_cols = [c for c in ("college_code","college_id","institute_code","college_name") if c in df.columns]
            if id_cols:
                # normalize comparable columns to strings
                for c in id_cols:
                    df[c] = df[c].astype(str)
                cand = df[
                    sum([df[c].isin(id_variants) for c in id_cols])
                    .astype(bool)
                ]
                # filter by round/quota/category where these cols exist
                if "round_code" in cand.columns and round_code:
                    cand = cand[cand["round_code"] == round_code]
                if "quota" in cand.columns and quota:
                    cand = cand[cand["quota"].isin(q_variants)]
                if "category" in cand.columns and category:
                    cand = cand[cand["category"].isin(c_variants)]
                for col in ("ClosingRank","closing","closing_rank","rank"):
                    if col in cand.columns and not cand[col].dropna().empty:
                        vals = [ _value_to_closing(v) for v in cand[col].dropna().tolist() ]
                        best = _best_of(vals)
                        if best is not None:
                            return best
    except Exception:
        pass

    # 2) Try dictionary (many shapes)
    d = lookup_dict or {}
    found: list = []

    def _parts_from_key(k):
        # tuple/list → list[str]
        if isinstance(k, (list, tuple)):
            return [str(x).strip() for x in k]
        # pipe-string
        if isinstance(k, str) and "|" in k:
            return [p.strip() for p in k.split("|") if p.strip()]
        # single
        return [str(k).strip()]

    def _key_matches(parts: list[str]):
        pset = {p for p in parts if p}
        # id present?
        has_id = any( (p in id_variants) or (_CODE_RE.match(p) and re.sub(r"\D+","",p) in id_variants)
                      for p in pset )
        if not has_id: return False
        # round (if we care)
        if round_code and not any(p == round_code for p in pset):
            # tolerate when round not encoded in key
            has_round = any(p == round_code for p in pset)
            if not has_round and any("R" in p for p in pset):
                return False
        # quota (tolerant)
        if quota and not any(p in q_variants for p in pset):
            # tolerate absence
            pass
        # category (tolerant)
        if category and not any(p in c_variants for p in pset):
            # tolerate absence
            pass
        return True

    try:
        # flat shapes
        for k, v in d.items():
            parts = _parts_from_key(k)
            if _key_matches(parts):
                cr = _value_to_closing(v)
                if cr is not None:
                    found.append(cr)

        # nested under identifiers
        for ide in id_variants:
            node = d.get(ide)
            if isinstance(node, dict):
                # try chains like node[round][quota][category] and loose forms
                for r_try in r_variants:
                    lvl_r = node.get(r_try) if (isinstance(node, dict) and r_try in node) else node
                    if lvl_r is None: continue
                    if not isinstance(lvl_r, dict):
                        cr = _value_to_closing(lvl_r)
                        if cr is not None: found.append(cr); continue
                    for q_try in q_variants:
                        lvl_q = lvl_r.get(q_try) if (isinstance(lvl_r, dict) and q_try in lvl_r) else lvl_r
                        if lvl_q is None: continue
                        if not isinstance(lvl_q, dict):
                            cr = _value_to_closing(lvl_q)
                            if cr is not None: found.append(cr); continue
                        for c_try in c_variants:
                            lvl_c = lvl_q.get(c_try) if (isinstance(lvl_q, dict) and c_try in lvl_q) else lvl_q
                            cr = _value_to_closing(lvl_c)
                            if cr is not None: found.append(cr)
            # lists under id
            if isinstance(node, list):
                for item in node:
                    cr = _value_to_closing(item)
                    if cr is not None: found.append(cr)
    except Exception:
        pass

    return _best_of(found) if found else None


def _best_of(vals: list):
    """Prefer numeric (smallest). Else first non-empty str."""
    best_num = None
    best_str = None
    for v in vals:
        if v is None: continue
        try:
            n = int(float(v))
            if best_num is None or n < best_num:
                best_num = n
        except Exception:
            if best_str is None:
                best_str = str(v)
    return best_num if best_num is not None else best_str



def _inst_type_from_row(r: dict) -> str:
    name = _safe_upper(r.get("college_name") or r.get("College Name"))
    own  = _safe_upper(r.get("ownership"))
    if "AIIMS" in name:  return "AIIMS (central govt)"
    if "JIPMER" in name: return "INI (central govt)"
    gov_tokens = ("GOV", "GOVT", "GOVERNMENT", "CENTRAL", "STATE", "NCT")
    if any(tok in own for tok in gov_tokens):
        return "Government medical college"
    if any(tok in own for tok in ("DEEMED","PRIVATE","TRUST","SOCIETY")):
        return "Private/Deemed university"
    # last-resort fee heuristic
    fee_raw = r.get("total_fee")
    try:
        fee_val = float(str(fee_raw).replace(",", ""))
    except Exception:
        fee_val = None
    if fee_val is not None and fee_val <= 100000:
        return "Government medical college"
    return own or "Medical college"

def _city_vibe_from_row(r: dict) -> str:
    city  = _safe_upper(r.get("city") or r.get("City"))
    state = _safe_upper(r.get("state") or r.get("State"))
    metro_cities = {"DELHI","NEW DELHI","MUMBAI","PUNE","BENGALURU","BANGALORE","HYDERABAD","CHENNAI","KOLKATA","AHMEDABAD"}
    metro_states = {"DELHI (NCT)","DELHI","MAHARASHTRA","KARNATAKA","TELANGANA","TAMIL NADU","WEST BENGAL"}
    if city in metro_cities or state in metro_states or "DELHI" in state:
        return "Metro pace; higher living costs; English/Hindi widely used"
    return "Calmer pace; mid living costs; local language common"

def get_openai_client():
    return _get_openai_client()

def _ensure_openai_client():
    # Only needed by ask_openai_vision/_gen_quick_qna if you use OpenAI SDK
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def _get_openai_client():
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = OpenAI()  # uses OPENAI_API_KEY env
    return _client_singleton


def _to_int(x):
    try:
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return None

def _to_fee_lakh(x):
    try:
        s = str(x).lower().replace(",", "").strip()
        if "l" in s:  # e.g., "7.3 L"
            return float(s.replace("l", "").strip())
        v = float(s)
        return v / 100000.0 if v > 99999 else v  # rupees → lakhs heuristic
    except Exception:
        return None

def _risk_label(closing, air):
    if closing is None or air is None:
        return None
    if closing >= air * 1.5:
        return "safe"
    if closing >= air * 1.1:
        return "moderate"
    return "dream"

def _notes_deterministic(facts, air):
    lines = ["*AI Notes (Top 10 from your list)*\n"]
    for f in facts:
        name = f.get("name") or "—"
        code = f.get("code") or "—"
        risk = _risk_label(f.get("closing_rank"), air)
        bullets = []
        if f.get("closing_rank") is not None:
            bullets.append(f"closing rank: {f['closing_rank']}")
        if f.get("nirf") is not None:
            bullets.append(f"NIRF: {f['nirf']}")
        fee = f.get("fee")
        if fee is not None:
            bullets.append(f"fee: ~{fee:.1f} L")
        owner = f.get("ownership")
        if owner:
            bullets.append(owner)
        st = f.get("state")
        if st:
            bullets.append(st)
        if f.get("hostel") is True:  # show only if TRUE
            bullets.append("hostel: yes")
        tag = f" — {risk}" if risk else ""
        lines.append(f"*{f['rank']}. {name}* (`{code}`){tag}\n- " + " • ".join(bullets) + "\n")
    lines.append("_Tip: safe = easier than your AIR, dream = tougher. Keep a 60–30–10 safe/moderate/dream mix._")
    return "\n".join(lines)

def _notes_via_llm(facts, air):
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception:
        return None

    compact = []
    for f in facts:
        item = {
            "rank": f["rank"],
            "name": f.get("name"),
            "code": f.get("code"),
            "state": f.get("state"),
            "closing": f.get("closing_rank"),
            "nirf": f.get("nirf"),
            "fee_lakh": f.get("fee"),
            "owner": f.get("ownership"),
            "hostel": (True if f.get("hostel") is True else None),
            "risk": _risk_label(f.get("closing_rank"), air),
        }
        compact.append({k: v for k, v in item.items() if v is not None})

    prompt = (
        "Write quick notes for a NEET shortlist. Use ONLY the given facts. "
        "Do not invent. Keep original order (by 'rank'). "
        "For each item output:\n"
        "Line 1: '**<rank>. <name>** (`<code>`) — <risk if present>'\n"
        "Line 2: fragments separated by ' • ' using available fields among: "
        "closing, NIRF, fee_lakh (format '~7.3 L'), owner, state, hostel (say 'hostel: yes').\n"
        "End with a brief generic tip on balancing safe/moderate/dream.\n\n"
        f"Candidate AIR: {air}\nFACTS JSON:\n{compact}\n\n"
        "Return markdown only."
    )
    try:
        resp = client.responses.create(model="gpt-4o-mini", input=prompt)
        txt = getattr(resp, "output_text", None)
        return txt.strip() if isinstance(txt, str) and txt.strip() else None
    except Exception:
        return None

def _yn(v):
    if v is None: return "ℹ️ verify"
    if isinstance(v, float) and str(v) == "nan": return "ℹ️ verify"
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"y","yes","true","1","available"}: return "✅"
        if s in {"n","no","false","0","not available"}: return "❌"
        if s in {"—","-","na","n/a",""}: return "ℹ️ verify"
    if v is True or v == 1:  return "✅"
    if v is False or v == 0: return "❌"
    return "ℹ️ verify"


def _truthy_or_none(x):
    """Parse many possible 'yes/no' encodings; None if unknown."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()

    TRUTHY = {
        "y","yes","true","t","1","available","avail","present","has","hostel yes","✔","✅","✓"
    }
    FALSY = {
        "n","no","false","f","0","na","n/a","not available","absent","none","nil","x","✖","✘","❌","no hostel","-"
    }
    if s in TRUTHY: return True
    if s in FALSY: return False
    if any(ch in s for ch in ("✅","✔","✓")): return True
    if any(ch in s for ch in ("❌","✖","✘")): return False
    try:
        return bool(int(s))
    except Exception:
        return None



def hostel_badge(row) -> str:
    """
    Robust hostel icon:
    - check several keys in the row
    - fallback to COLLEGES meta by college_code
    - return ❓ if unknown (don’t default to ❌)
    """
    # direct keys on the row
    for k in ("hostel_available", "hostel", "Hostel", "has_hostel"):
        if k in row:
            v = _truthy_or_none(row[k])
            if v is True:  return "✅"
            if v is False: return "✅"

    # fallback: look in COLLEGES by code
    code = row.get("college_code") or row.get("code") or row.get("college_id")
    try:
        if code and isinstance(COLLEGES, pd.DataFrame) and not COLLEGES.empty and "college_code" in COLLEGES.columns:
            sub = COLLEGES[COLLEGES["college_code"].astype(str) == str(code)]
            if not sub.empty:
                meta_val = None
                for mkey in ("hostel_available","Hostel","hostel","has_hostel"):
                    if mkey in sub.columns:
                        meta_val = sub.iloc[0][mkey]
                        break
                v = _truthy_or_none(meta_val)
                if v is True:  return "✅"
                if v is False: return "✅"
    except Exception:
        pass

    return "❓"

def _notes_strip_markdown(text: str) -> str:
    t = str(text or "")
    t = re.sub(r'^\s{0,3}#{1,6}\s*', '', t, flags=re.M)
    t = re.sub(r'(\*\*|__)(.+?)(\*\*|__)\s*:?$', r'\2', t, flags=re.M)
    t = re.sub(r'(\*\*|__)(.+?)\1', r'\2', t, flags=re.S)
    t = re.sub(r'(\*|_)(.+?)\1', r'\2', t, flags=re.S)
    t = re.sub(r'`([^`]+)`', r'\1', t)
    t = re.sub(r'^\s*[-*]|\s*\d+\.', '•', t, flags=re.M)
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'[ \t]+$', '', t, flags=re.M)
    return t.strip()

async def send_ai_notes(bot, chat_id: int, text: str):
    msg = _notes_strip_markdown(text)
    if not msg:
        msg = "No notes available."
    chunk = 3500
    for i in range(0, len(msg), chunk):
        await bot.send_message(chat_id=chat_id, text=msg[i:i+chunk], parse_mode=None, disable_web_page_preview=True)


        
def _cleanup_latex(s: str) -> str:
    # remove display math markers
    s = re.sub(r"\\\[|\\\]", "", s)
    # remove inline math markers
    s = re.sub(r"\\\(|\\\)", "", s)
    # replace \cdot with ×
    s = s.replace(r"\cdot", "×")
    # strip \text{...}
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    # remove extra backslashes
    s = s.replace("\\", "")
    return s.strip()

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    KeyboardButton,
    constants,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("aceit-bot")


# ---------- Env ----------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN not found in .env")

# ---------- Constants/paths ----------

QUIZ_JSON_PATH = os.getenv("QUIZ_JSON_PATH", str(Path(__file__).parent / "quiz.json"))




OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


QUIZ_SESSIONS: Dict[int, Dict[str, Any]] = {}          # user_id -> {"questions":[], "answers":{}, "index": int}

# difficulty mapping used by /quiz5medium etc.
_DIFF_MAP = {1: "low", 2: "medium", 3: "high"}

DATA_DIR = "data"
EXCEL_PATH = "MCC_Final_with_Cutoffs_2024_2025.xlsx"  # your file
ACTIVE_CUTOFF_ROUND_DEFAULT = "2025_R1"
TG_LIMIT = 4000
NEET_CANDIDATE_POOL_DEFAULT = 2300000  # adjust if you want
_ALLOWED_ANY_QUOTAS = ["AIQ", "Deemed", "Central" ]
CUTSHEET_OVERRIDE = {"2025_R1": None, "2024_Stray": None}
ASK_MOCK_RANK, ASK_MOCK_SIZE = range(307, 309)

COACH_TOP_N = 40         # how many rows to send to LLM
COACH_SHOW_N = 12        # how many to display back to user
COACH_MODEL = "gpt-4o-mini"  # or your preferred small JSON-capable model

COACH_ADJUST = "coach_adjust"   # callback query data prefixes
COACH_SAVE   = "coach_save"

MENU_COACH = "menu_coach"

NOTES_TOP_N = 10
GENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # or gpt-4.1-mini

SAFE_CUTOFF = 0.90   # AIR <= 0.90 * ClosingRank → "safe"
DREAM_CUTOFF = 1.10  # AIR >  1.10 * ClosingRank → "dream"



COACH_TOP_N  = 200     # pool to send to LLM/fallback
COACH_SHOW_N = 20      # lines to show in Telegram (avoid 4096 char limit)
OPENAI_MODEL = "gpt-4o-mini"

df_candidate = None


QUIZ_JSON = os.environ.get("QUIZ_JSON", "").strip()

# ---------------------- Canonical helpers (use once, globally) ----------------------
import re


from dataclasses import dataclass
import json, os, math, time
from typing import List, Dict, Any

@dataclass
class CoachPlan:
    ordered_codes: List[str]
    rationales: Dict[str, str]
    risk_mix: Dict[str, int]

def _trim(s: str, n: int = 140) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"

async def _safe_clear_kb(q):
    """
    Remove the inline keyboard from the message that triggered this callback.
    Silently ignore benign 400s like 'message is not modified' or 'not found'.
    """
    try:
        await q.edit_message_reply_markup(reply_markup=None)
    except BadRequest as e:
        msg = str(e)
        if ("message is not modified" in msg
            or "message to edit not found" in msg
            or "MESSAGE_ID_INVALID" in msg):
            return
        # Log uncommon reasons for visibility
        log.warning("editMessageReplyMarkup(clear) failed: %s", msg)

async def _safe_set_kb(q, kb):
    """
    Replace the inline keyboard on the message that triggered this callback.
    """
    try:
        await q.edit_message_reply_markup(reply_markup=kb)
    except BadRequest as e:
        msg = str(e)
        if ("message is not modified" in msg
            or "message to edit not found" in msg
            or "MESSAGE_ID_INVALID" in msg):
            return
        log.warning("editMessageReplyMarkup(set) failed: %s", msg)


async def _safe_clear_markup(query):
    try:
        await query.edit_message_reply_markup(None)
    except BadRequest:
        pass
    except Exception:
        pass

def _to_int(x):
    try:
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return None

def _to_float(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None

def _to_fee(x):
    # returns float in lakhs if possible, else None
    try:
        s = str(x).lower().replace(",", "").strip()
        # tolerate "7.3 L" or "730000"
        if "l" in s:
            return float(s.replace("l", "").strip())
        v = float(s)
        # heuristically convert to lakhs if looks like rupees
        return v / 100000.0 if v > 99999 else v
    except Exception:
        return None

def _risk_label(closing, air):
    if closing is None or air is None:
        return None
    # lower closing => better college; compare with candidate AIR
    if closing >= air * 1.5:
        return "safe"
    if closing >= air * 1.1:
        return "moderate"
    return "dream"

def _notes_deterministic(facts, air):
    """Plain notes without LLM; uses only known fields."""
    lines = ["*AI Notes (Top 10 from your list)*\n"]
    for f in facts:
        name = f.get("name") or "—"
        code = f.get("code") or "—"
        risk = _risk_label(f.get("closing_rank"), air)
        bullets = []
        if f.get("closing_rank") is not None:
            bullets.append(f"closing rank: {f['closing_rank']}")
        if f.get("nirf") is not None:
            bullets.append(f"NIRF: {f['nirf']}")
        fee = f.get("fee")
        if fee is not None:
            bullets.append(f"fee: ~{fee:.1f} L")
        owner = f.get("ownership")
        if owner:
            bullets.append(owner)
        st = f.get("state")
        if st:
            bullets.append(st)
        # Show hostel only if TRUE; skip if False/None to avoid noisy ❌
        if f.get("hostel") is True:
            bullets.append("hostel: yes")
        tag = f" • {risk}" if risk else ""
        lines.append(f"*{f['rank']}. {name}* (`{code}`){tag}\n- " + " • ".join(bullets) + "\n")
    lines.append("_Tip: safe = easier than your AIR, dream = tougher. Balance 60–30–10 safe/moderate/dream for counselling._")
    return "\n".join(lines)

def _notes_via_llm(facts, air):
    """Ask GPT for short notes; returns markdown string or None on failure."""
    # keep prompt tiny; never invent unknowns; keep original order
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception:
        return None

    # Build compact payload; skip unknowns
    compact = []
    for f in facts:
        item = {
            "rank": f["rank"],
            "name": f.get("name"),
            "code": f.get("code"),
            "state": f.get("state"),
            "closing": f.get("closing_rank"),
            "nirf": f.get("nirf"),
            "fee_lakh": f.get("fee"),
            "owner": f.get("ownership"),
            "hostel": (True if f.get("hostel") is True else None),  # only include if True
            "risk": _risk_label(f.get("closing_rank"), air),
        }
        compact.append({k: v for k, v in item.items() if v is not None})

    prompt = (
        "You are helping a NEET counselling bot write quick notes for a shortlist. "
        "Use ONLY the given facts. Do not guess or add external info. "
        "Keep original order (by 'rank'). For each college, output 1-2 short lines: "
        "Line 1: '**<rank>. <name>** (`<code>`) — <risk if present>'. "
        "Line 2: bullet-like fragments separated by ' • ' using available fields among: "
        "closing, NIRF, fee_lakh (format '~7.3 L'), owner, state, hostel (say 'hostel: yes'). "
        "After all items, add one line with a generic tip about balancing safe/moderate/dream choices.\n\n"
        f"Candidate AIR (optional): {air}\n"
        f"FACTS JSON:\n{compact}\n\n"
        "Return markdown text only."
    )

    try:
        resp = client.responses.create(model="gpt-4o-mini", input=prompt)
        text = resp.output_text
        if text and isinstance(text, str) and len(text.strip()) > 0:
            return text.strip()
        return None
    except Exception:
        return None

def _as_records(df_or_list):
    if hasattr(df_or_list, "to_dict"):
        return df_or_list.to_dict("records")
    return list(df_or_list or [])

def _is_missing(v):
    return v is None or str(v).strip().lower() in {"", "—", "na", "n/a", "nan"}

def _num_or_inf(v):
    try:
        s = str(v).replace(",", "").strip()
        if not s:
            return float("inf")
        return float(s)
    except Exception:
        return float("inf")

def _fmt_rank_val(v):
    try:
        s = _safe_str(v, "")
        return "—" if not s else f"{int(float(s)):,}"
    except Exception:
        return "—"

def _fmt_money(v):
    try:
        s = str(v).replace(",", "").strip()
        if not s:
            return "—"
        n = float(s)
        return f"₹{int(n):,}"
    except Exception:
        return "—"

def _pick(d: dict, *keys):
    for k in keys:
        if d.get(k) not in (None, "", "—"):
            return d.get(k)
    return None

def _format_row_multiline(r: dict, user: dict, df_lookup=None) -> str:
    """Name, place; then Closing Rank and Annual Fee each on its own line. No 'm' fallbacks here."""
    # NaN/None safe strings
    name  = _safe_str(_pick(r, "college_name", "College Name")) or "—"
    city  = _safe_str(_pick(r, "city", "City"))
    state = _safe_str(_pick(r, "state", "State"))
    place = ", ".join([x for x in (city, state) if x])

    # rank lookup context
    round_ui = (user or {}).get("cutoff_round") or (user or {}).get("round") or "2025_R1"
    quota    = (user or {}).get("quota") or "AIQ"
    category = (user or {}).get("category") or "General"

    # allow a pre-attached df lookup
    try:
        df_lookup = r.get("_df_lookup") or df_lookup
    except Exception:
        pass

    # closing rank: prefer inline fields, else compute from identifiers
    closing = r.get("ClosingRank") or r.get("closing") or r.get("rank")
    if _is_missing(closing):
        ids = [
            r.get("college_code"), r.get("code"),
            r.get("college_id"), r.get("institute_code"),
            _pick(r, "college_name", "College Name"),
        ]
        closing = _closing_rank_for_identifiers(
            [x for x in ids if not _is_missing(x)],
            round_ui, quota, category,
            df_lookup=df_lookup, lookup_dict=CUTOFF_LOOKUP
        )

    fee = _pick(r, "total_fee", "Fee")

    header = f"{name}" + (f", {place}" if place else "")
    cr_ln  = f"Closing Rank { _fmt_rank_val(closing) }"
    fee_ln = f"Annual Fee { _fmt_money(fee) }"
    return "\n".join([header, cr_ln, fee_ln])

def _deemed_only(rows):
    out = []
    for r in rows:
        own = str(r.get("ownership") or "").lower()
        if "deemed" in own:  # strict deemed filter
            out.append(r)
    return out

def _sorted_deemed_by_fee(colleges, limit=10):
    rows = _as_records(colleges)
    rows = _deemed_only(rows)
    # keep MBBS if you store other courses in same sheet
    rows = [r for r in rows if str(r.get("course") or "MBBS").strip().upper() == "MBBS"]
    rows.sort(key=lambda x: _num_or_inf(x.get("total_fee") or x.get("Fee")))
    return rows[:limit] 

def _norm_row_for_cache(r: dict) -> dict:
    """Normalize one predictor row for caching/AI notes without changing order."""
    code = (
        r.get("college_code")
        or r.get("code")
        or r.get("College Code")
        or r.get("college_id")
    )
    name = r.get("college_name") or r.get("College Name") or r.get("name")
    state = r.get("state") or r.get("State")

    closing = (
        r.get("ClosingRank")
        or r.get("closing")
        or r.get("closing_rank")
        or r.get("rank")
    )
    try:
        closing = int(float(closing))
    except Exception:
        closing = None

    return {
        "college_code": str(code) if code is not None else None,
        "college_name": name,
        "state": state,
        "ClosingRank": closing,
        "nirf_rank_medical_latest": r.get("nirf_rank_medical_latest") or r.get("NIRF"),
        "total_fee": r.get("total_fee") or r.get("Fee") or r.get("Total Fee"),
        "ownership": r.get("ownership") or r.get("Ownership"),
        "website": r.get("website") or r.get("Website"),
    }


async def start(update, context):
    await update.message.reply_text("Hello from Aceit!")





    

async def coach_notes_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Render AI notes for the top 10 from the last shortlist; no reordering."""
    q = update.callback_query
    await q.answer()
    ud = context.user_data or {}
    shortlist = ud.get("last_predict_shortlist") or []
    air = ud.get("last_predict_air")

    if not shortlist:
        await q.edit_message_text("No shortlist in memory. Please run /predict first.")
        return

    # build facts for top 10, preserving order
    facts = []
    for i, r in enumerate(shortlist[:10], 1):
        facts.append({
            "rank": i,
            "code": r.get("college_code"),
            "name": r.get("college_name"),
            "state": r.get("state"),
            "closing_rank": _to_int(r.get("ClosingRank")),
            "nirf": _to_int(r.get("nirf_rank_medical_latest")),
            "fee": _to_fee(r.get("total_fee")),
            "ownership": r.get("ownership"),
            # only True means we know hostel is available; otherwise omit
            "hostel": (True if r.get("hostel_available") is True else None),
        })

    # Try LLM first; then fallback
    text = _notes_via_llm(facts, air)
    if not text:
        text = _notes_deterministic(facts, air)

    # Don’t edit the original shortlist; send a fresh message
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        parse_mode="Markdown",
        disable_web_page_preview=True,
    )



def _get_cutoffs_df_from_context(context):
    df = globals().get("CUTOFFS_DF", None)
    if df is None or getattr(df, "empty", True):
        df = context.application.bot_data.get("CUTOFFS_DF")
    return df

       

def _risk_bucket(closing_rank: int | float | None, air: int) -> str:
    if closing_rank in (None, "", 0):
        return "moderate"
    try:
        r = air / float(closing_rank)
    except Exception:
        return "moderate"
    if r <= SAFE_CUTOFF:
        return "safe"
    if r > DREAM_CUTOFF:
        return "dream"
    return "moderate"

def _get_cutoffs_df_from_context(context):
    df = globals().get("CUTOFFS_DF")
    if df is None or getattr(df, "empty", True):
        df = context.application.bot_data.get("CUTOFFS_DF")
    return df

def _norm_quota(q: str | None) -> str | None:
    if not q: return None
    q = str(q).strip()
    return {"Central": "AIQ", "All India Quota": "AIQ"}.get(q, q)

def _norm_category(c: str | None) -> str | None:
    if not c: return None
    c = str(c).strip()
    return {"General": "OP", "Open": "OP", "UR": "OP"}.get(c, c)

def _mk_candidate_payload(df, air: int) -> List[Dict[str, Any]]:
    # keep only needed fields; be defensive with .get
    fields = {
        "college_code", "college_name", "state", "city",
        "ownership", "total_fee", "hostel_available",
        "ClosingRank", "category", "quota"
    }
    out = []
    for _, row in df.iterrows():
        obj = {k: row.get(k) if hasattr(row, "get") else row[k] for k in row.index if k in fields}
        # normalize types
        try:
            obj["ClosingRank"] = int(obj.get("ClosingRank", 0)) if obj.get("ClosingRank") not in (None, "", "—") else None
        except Exception:
            obj["ClosingRank"] = None
        obj["risk"] = _risk_bucket(obj.get("ClosingRank"), air)
        obj["college_code"] = str(obj.get("college_code") or obj.get("college_id") or "")
        out.append(obj)
    return out

def _fallback_rank(cands: List[Dict[str, Any]], air: int) -> CoachPlan:
    # bucket → sort by tougher college first (smaller ClosingRank)
    buckets = {"safe": [], "moderate": [], "dream": []}
    for row in cands:
        row = dict(row)
        cr = row.get("ClosingRank")
        row["_bucket"] = _risk_bucket(cr, air)
        buckets[row["_bucket"]].append(row)

    for b in buckets.values():
        b.sort(key=lambda x: (x.get("ClosingRank") or 9_999_999))

    ordered = buckets["safe"] + buckets["moderate"] + buckets["dream"]
    ordered_codes = [str(x.get("college_code")) for x in ordered if x.get("college_code")]

    # simple built-in rationales
    rats = {}
    for x in ordered[:COACH_SHOW_N]:
        bits = []
        if x.get("state"): bits.append(x["state"])
        if "total_fee" in x and pd.notna(x["total_fee"]): bits.append(f"fee {x['total_fee']}")
        if "ownership" in x: bits.append(x["ownership"])
        rats[str(x["college_code"])] = " • ".join(bits) or "fit based on last year closing"
    return CoachPlan(ordered_codes=ordered_codes, rationales=rats,
                     risk_mix={"safe": 40, "moderate": 40, "dream": 20})

def _mk_ai_prompt(cands: List[Dict[str, Any]], air: int) -> str:
    # keep it tiny; no external fetch, just reasoning over provided fields
    lines = [
        "You are counselling assistant for NEET MBBS preferences.",
        "Reorder the colleges into a single preference list (max 20).",
        "Use only the given fields; DO NOT invent facts.",
        f"Applicant AIR: {air}",
        "",
        "Fields per college: college_code, college_name, state, ClosingRank, nirf_rank_medical_latest, total_fee, hostel_available, ownership.",
        "Heuristic: prefer tougher colleges (smaller ClosingRank) within the same risk bucket; bucket by AIR vs ClosingRank.",
        "Return JSON with keys: ordered_codes (list of college_code) and rationales (dict code->short reason). Keep reasons to one line.",
        "",
        "CANDIDATES:"
    ]
    for c in cands[:50]:
        lines.append(str({k: c.get(k) for k in [
            "college_code","college_name","state","ClosingRank",
            "nirf_rank_medical_latest","total_fee","hostel_available","ownership"
        ]}))
    lines.append("\nOutput:")
    return "\n".join(lines)



def _render_coach_plan(plan: CoachPlan, cands: List[Dict[str, Any]], air: int):
    by_code = {str(x["college_code"]): x for x in cands} 
    lines = ["*Your AI Coach Preference List*"]
    lines.append(f"_Mix:_ {plan.risk_mix.get('safe',0)}% safe · "
                 f"{plan.risk_mix.get('moderate',0)}% moderate · "
                 f"{plan.risk_mix.get('dream',0)}% dream\n")

    rank = 1
    for code in plan.ordered_codes:
        row = by_code.get(code) or {}
        nm = row.get("college_name", "?")
        rk_val = row.get("ClosingRank", "—")

        state  = row.get("state", "")
        fee    = row.get("total_fee", row.get("Fee", "?"))
        hostel = hostel_badge(row)

        rz = plan.rationales.get(code)
        if not rz:
            rz = _trim(f"{state} • fee {fee} • hostel {hostel}")

        risk = _risk_bucket(row.get("ClosingRank"), int(air)).upper()

        lines.append(
            f"*{rank}. {nm}* (`{code}`)\n"
            f"    {rz}\n"
            f"    _closing {rk_val} • {risk}_"
        )
        rank += 1
        if rank > COACH_SHOW_N:
           break
    text = "\n".join(lines)
    kb = [
        [InlineKeyboardButton("💾 Save as My List", callback_data=f"{COACH_SAVE}:v1")],
        [
            InlineKeyboardButton("Make Safer", callback_data=f"{COACH_ADJUST}:safer"),
            InlineKeyboardButton("Balanced",   callback_data=f"{COACH_ADJUST}:balanced"),
            InlineKeyboardButton("More Dream", callback_data=f"{COACH_ADJUST}:dreamier"),
        ],
    ]
    return text, InlineKeyboardMarkup(kb)

async def coach_adjust_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    mode = q.data.split(":")[1]  # safer|balanced|dreamier
    uid = update.effective_user.id if update.effective_user else "anon"
    cache = context.application.bot_data.get("AI_COACH_CACHE", {}).get(uid)
    if not cache:
        await q.edit_message_text("No coach session found. Run /coach after /predict.")
        return

    cands = cache["candidates"]
    prefs = cache["prefs"]

    if mode == "safer":    prefs["target_mix"] = {"safe": 60, "moderate": 30, "dream": 10}
    if mode == "balanced": prefs["target_mix"] = {"safe": 40, "moderate": 40, "dream": 20}
    if mode == "dreamier": prefs["target_mix"] = {"safe": 25, "moderate": 45, "dream": 30}

    air = context.user_data.get("rank_air") or 1
    try:
        plan = _call_ai_coach(cands, prefs)
    except Exception:
        # fallback: reshuffle by risk buckets according to target mix
        df = pd.DataFrame(cands)
        plan = _fallback_rank(df, int(air))

    cache["plan"] = plan.__dict__
    cache["prefs"] = prefs

    text, kb = _render_coach_plan(plan, cands, int(air))
    # Telegram 4096 hard limit safety
    text = text[:4000]
    await q.edit_message_text(text, parse_mode="Markdown", reply_markup=kb)

async def coach_save_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = update.effective_user.id if update.effective_user else "anon"
    cache = context.application.bot_data.get("AI_COACH_CACHE", {}).get(uid)
    if not cache:
        await q.edit_message_text("Nothing to save yet. Run /coach first.")
        return
    context.user_data["my_pref_list"] = cache  # simple per-user stash
    await q.edit_message_text("✅ Saved! You can retrieve it anytime with /mylist")

async def mylist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tgt = _target(update)
    data = context.user_data.get("my_pref_list")
    if not data:
        await tgt.reply_text("No saved list yet. Use AI Coach and tap *Save*.", parse_mode="Markdown")
        return
    plan = CoachPlan(**data["plan"])
    text, kb = _render_coach_plan(plan, data["candidates"], int(context.user_data.get("rank_air") or 1))
    await tgt.reply_text(text[:4000], parse_mode="Markdown", reply_markup=kb)




def _norm_key(x: str | None) -> str:
    """Normalize codes/IDs like 'C0004', 'AIIMS-DELHI' -> 'C0004', 'AIIMSDELHI'."""
    return re.sub(r"[^A-Z0-9]+", "", (str(x).strip().upper() if x else ""))

def _name_key(x: str | None) -> str:
    """Normalize display names for fuzzy matching (lowercase alnum)."""
    return re.sub(r"[^a-z0-9]+", "", (str(x).strip().lower() if x else ""))

def _safe_int(v):
    """Coerce values like '1,234', '  —  ', 'N/A', 123.0 to int or None."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.upper() in {"NA", "N/A", "NONE", "—", "-", "NULL"}:
        return None
    s = s.replace(",", "")
    try:
        # handle floats-in-excel like 48.0
        return int(float(s))
    except Exception:
        return None

def _cleanup_latex(s: str) -> str:
    if not isinstance(s, str):
        return s

    # remove display math markers \[ ... \]
    s = re.sub(r"\\\[|\\\]", "", s)

    # remove inline math markers \( ... \)
    s = re.sub(r"\\\(|\\\)", "", s)

    # strip \text{...} → ...
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)

    # common LaTeX operators
    s = s.replace(r"\cdot", "×")
    s = s.replace(r"\times", "×")
    s = s.replace(r"\le", "≤").replace(r"\ge", "≥")
    s = s.replace(r"\neq", "≠")
    s = s.replace(r"\approx", "≈")

    # frac: \frac{a}{b} → a/b   (keeps it readable in Telegram)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", s)

    # superscripts/subscripts (very basic)
    s = re.sub(r"\^\{([^}]*)\}", r"^\1", s)
    s = re.sub(r"_\{([^}]*)\}", r"_\1", s)

    # remove stray backslashes left over
    s = s.replace("\\", "")

    # collapse extra whitespace
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def _canon_quota(q: str | None) -> str:
    """
    Canonicalize quota labels to buckets used everywhere else:
      - AIQ / All India / 15%         -> 'AIQ'
      - State / Home / 85%            -> 'State'
      - Central / GOI / DGHS / CU     -> 'Central'
      - Deemed / Management / Paid    -> 'Deemed'
      - NRI / OCI / PIO               -> 'NRI'
      - Fallback: Title Case or 'AIQ'
    """
    s = (q or "").strip().upper()
    s = re.sub(r"\s+", " ", s)

    if s in {"AIQ", "ALL INDIA", "ALLINDIA", "15%", "15 PERCENT"}:
        return "AIQ"
    if s in {"STATE", "SQ", "HOME", "DOMICILE", "85%", "85 PERCENT"}:
        return "State"
    # Central / GOI bucket
    if s in {"GOI", "CENTRAL", "DGHS", "CU", "CENTRAL UNIVERSITY", "CENTRAL UNIV"}:
        return "Central"
    # Deemed / Management bucket
    if ("DEEMED" in s) or ("MANAGEMENT" in s) or ("PAID" in s):
        return "Deemed"
    # NRI etc.
    if s in {"NRI", "OCI", "PIO"}:
        return "NRI"

    return s.title() if s else "AIQ"

def _canon_cat(v: object) -> str:
    """
    Canonicalize category labels to one of:
      "General", "EWS", "OBC", "SC", "ST",
      and PwD versions: "General_PwD", "EWS_PwD", ...
    Accepts a wide range of tokens commonly found in sheets.
    """
    s = str(v or "").strip().upper()

    # PwD flags attached to category (e.g., "UR-PWD", "OBC PwBD", "GEN PH")
    is_pwd = any(t in s for t in ("PWD", "PWBD", "PH"))

    # core buckets
    if any(t in s for t in ("UR", "GEN", "GENERAL", "OPEN", "OP", "OC", "UNRESERVED")):
        base = "General"
    elif "EWS" in s:
        base = "EWS"
    elif any(t in s for t in ("OBC", "BC")):      # <-- IMPORTANT: BC → OBC
        base = "OBC"
    elif "SC" in s:
        base = "SC"
    elif "ST" in s:
        base = "ST"
    else:
        # keep last resort so we don't drop rows silently
        base = s.title() if s else ""

    if not base:
        return ""

    return f"{base}_PwD" if is_pwd else base


def _cat_aliases(base_cat: str) -> list[str]:
    """
    Return likely aliases to try when matching columns/keys in wide sheets.
    """
    c = _canon_cat(base_cat)
    if c == "General":
        return ["General", "GEN", "UR", "OPEN", "OP", "OC"]
    if c == "EWS":
        return ["EWS"]
    if c == "OBC":
        return ["OBC", "OBC-NCL", "OBCNCL", "BC", "SEBC"]
    if c == "SC":
        return ["SC", "S.C", "SC(P)", "SC-PWD", "SC PH", "SC (NON-PWD)"]
    if c == "ST":
        return ["ST", "S.T", "ST(H)", "STH", "ST-PWD", "ST PH", "ST (NON-PWD)"]
    return [c]

# (Optional tiny wrapper kept for compatibility if you call _canon_category somewhere)
def _canon_category(v: str) -> str:
    return _canon_cat(v)
# -----------------------------------------------------------------------------------


def _coach_bucket(closing: int | None, air: int | None) -> str:
    if closing is None or air is None:
        return "?"
    # lower closing rank means better college; compare relative to AIR
    # Treat “safe” if closing is much higher (i.e., easier to get than your AIR)
    # tweak thresholds as you like
    if closing >= air * 1.5:
        return "SAFE"
    if closing >= air * 1.1:
        return "MODERATE"
    return "DREAM"

def _coach_make_plan_from_shortlist(cands: list[dict], air: int | None):
    """
    Sort candidates deterministically into a reasonable preference order:
    - Primarily by ClosingRank ascending (better college first)
    - Then by NIRF (lower is better)
    - Then by fee ascending
    Returns (ordered_codes, rationales_dict)
    """
    def _key(x):
        cr = x.get("ClosingRank")
        nr = x.get("nirf_rank_medical_latest")
        fe = x.get("total_fee")
        try: cr = int(cr)
        except Exception: cr = 10**9
        try: nr = int(nr)
        except Exception: nr = 10**9
        try: fe = float(str(fe).replace(",",""))
        except Exception: fe = 10**12
        return (cr, nr, fe)

    ordered = sorted([c for c in cands if c.get("college_code")], key=_key)
    ordered_codes = [c["college_code"] for c in ordered]

    # build short deterministic rationales
    raz = {}
    for r in ordered:
        parts = []
        if r.get("nirf_rank_medical_latest"):
            parts.append(f"NIRF {r['nirf_rank_medical_latest']}")
        if r.get("total_fee"):
            parts.append(f"fee {r['total_fee']}")
        if "hostel_available" in r:
            parts.append(f"hostel {'✅' if r['hostel_available'] else '❌'}")
        if r.get("state"):
            parts.append(r["state"])
        raz[r["college_code"]] = " • ".join(parts) if parts else "Good fit based on cutoffs"
    return ordered_codes, raz

def _coach_fallback_reason(row: dict) -> str:
    bits = []
    if row.get("state"): bits.append(row["state"])
    if row.get("total_fee"): bits.append(f"fee {row['total_fee']}")
    if "hostel_available" in row: bits.append(f"hostel {'✅' if row['hostel_available'] else '❌'}")
    return " • ".join(bits) if bits else "Good fit based on closing rank"

def _coach_llm_rationales(cands: list[dict], max_items: int = 20) -> dict[str, str] | None:
    """
    Optional: ask the LLM for 1-line reasons. If OPENAI_API_KEY is not set or
    call fails, return None and we’ll stick to deterministic reasons.
    """
    try:
        from openai import OpenAI  # uses env OPENAI_API_KEY
        client = OpenAI()
    except Exception:
        return None

    # keep prompt small
    small = []
    for r in cands[:max_items]:
        small.append({
            "name": r.get("college_name"),
            "state": r.get("state"),
            "closing": r.get("ClosingRank"),
            "nirf": r.get("nirf_rank_medical_latest"),
            "fee": r.get("total_fee"),
            "hostel": r.get("hostel_available"),
        })

    prompt = (
        "You are helping a NEET counselling bot. For each college below, return a terse, "
        "one-line reason (<=18 words) tailored to a general NEET candidate in India. "
        "Prefer objective points: closing rank competitiveness, NIRF, fees (lower is better), "
        "hostel availability, public vs private, and state context.\n\n"
        f"INPUT JSON:\n{small}\n\n"
        "Return JSON object mapping code -> reason. No extra text."
    )

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        text = resp.output_text
        import json
        j = json.loads(text)
        if isinstance(j, dict):
            # keep only strings
            return {k: (v if isinstance(v, str) else "") for k, v in j.items()}
        return None
    except Exception:
        return None

def _get_cutoffs_df_from_context(context) -> Optional["pd.DataFrame"]:
    df = context.application.bot_data.get("CUTOFFS_DF")
    if df is None:
        df = globals().get("CUTOFFS_DF")
    return df



def _trim(s: str, n: int = 140) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"

def _mk_candidate_payload(df, air: int) -> List[Dict[str, Any]]:
    cols = {
        "college_code", "college_name", "state", "ownership",
        "total_fee", "bond_years", "bond_penalty_lakhs",
        "nirf_rank_medical_latest", "hostel_available",
        "ClosingRank"  # ensure your predict output has this name
    }
    ex = []
    for _, r in df.iterrows():
        d = {k: r.get(k) for k in cols if k in df.columns}
        d["risk"] = _risk_bucket(d.get("ClosingRank") or None, air)
        # sanitize / types
        d["total_fee"] = None if pd.isna(d.get("total_fee")) else float(d.get("total_fee"))
        d["bond_years"] = None if pd.isna(d.get("bond_years")) else int(d.get("bond_years"))
        d["bond_penalty_lakhs"] = None if pd.isna(d.get("bond_penalty_lakhs")) else float(d.get("bond_penalty_lakhs"))
        d["nirf_rank_medical_latest"] = None if pd.isna(d.get("nirf_rank_medical_latest")) else int(d.get("nirf_rank_medical_latest"))
        d["hostel_available"] = str(d.get("hostel_available") or "").strip().lower() in ("y", "yes", "true", "1")
        ex.append(d)
    return ex[:COACH_TOP_N]

# pydantic schema for LLM JSON
class CoachPlan(BaseModel):
    ordered_codes: List[str] = Field(default_factory=list)
    risk_mix: Dict[str, int] = Field(default_factory=dict)   # {"safe":40,"moderate":40,"dream":20}
    rationales: Dict[str, str] = Field(default_factory=dict)



def _ai_client():
    # reuse your singleton client if you already have one
    from openai import OpenAI
    return OpenAI()  # gets OPENAI_API_KEY from env

def _call_ai_coach(cands: List[Dict[str, Any]], air: int) -> CoachPlan:
    try:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from env
        prompt = _mk_ai_prompt(cands, air)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # or your deployed model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        import json, re
        jtxt = re.search(r"\{.*\}", text, re.S)
        data = json.loads(jtxt.group(0)) if jtxt else json.loads(text)
        ordered = [str(x) for x in data.get("ordered_codes", [])]
        rats = {str(k): str(v) for k, v in (data.get("rationales") or {}).items()}
        if not ordered:
            raise RuntimeError("empty AI result")
        return CoachPlan(ordered_codes=ordered, rationales=rats,
                         risk_mix={"safe": 40, "moderate": 40, "dream": 20})
    except Exception:
        # fall back silently
        return _fallback_rank(cands, air)
    



ALLOWED_TAGS = {"b", "i", "u", "br"}
_BOLD_MD = re.compile(r"\*\*(.+?)\*\*")  # non-greedy

def _to_safe_html(raw: str) -> str:
    """
    Turn LLM text into Telegram-safe HTML:
    - Convert **bold** -> <b>bold</b> once.
    - Escape everything else.
    - Preserve only <b> tags.
    """
    if not raw:
        return ""
    s = raw.replace("\r\n", "\n").replace("\r", "\n")

    # 1) convert markdown bold exactly once
    s = _BOLD_MD.sub(r"<b>\1</b>", s)

    # 2) escape all HTML
    s = html.escape(s, quote=False)

    # 3) unescape allowed tags (<b> only here; add more if you later need)
    s = (s.replace("&lt;b&gt;", "<b>")
           .replace("&lt;/b&gt;", "</b>"))

    # 4) remove code fences/backticks and math delimiters if any slipped through
    s = re.sub(r"`{1,3}", "", s)
    s = s.replace("\\[", "").replace("\\]", "")

    # 5) trim
    return s.strip()


def _dedupe_results(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        nm = (r.get("college_name") or "").strip()
        key = _name_key(nm) if nm else None
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _tg_html(s: str) -> str:
    """
    Make HTML safe for Telegram:
    - <br> -> newline (also handles &lt;br&gt;)
    - collapse big gaps
    - drop any tags except a tiny whitelist (b only here)
    - fix stray 'b' artifacts before/after <b>…</b>
    """
    if not s:
        return ""
    # normalize <br> and escaped <br>
    s = re.sub(r"(?i)(<br\s*/?>|&lt;br\s*/?&gt;)", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    # strip any tags except <b>…</b>
    s = re.sub(r"(?is)</?(?!b\b)[a-z][^>]*>", "", s)

    # clean artifacts like "b <b>Title</b>" and "</b> b"
    s = re.sub(r"(^|\s)b\s*(?=<b>)", r"\1", s)
    s = re.sub(r"(?<=</b>)\s*b(\s|$)", r"\1", s)

    return s.strip()

def _tg_chunks(s: str, n: int = 3800):
    s = s or ""
    for i in range(0, len(s), n):
        yield s[i:i+n]

def _strip_html(s: str) -> str:
    """Turn simple HTML back into plain text for prompting."""
    s = s or ""
    s = re.sub(r"<\s*br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()



def _get_close_rank_from_rec(rec: dict, category: str):
    """
    rec example: {"General": 48, "EWS": 123, "OBC": 500, "General_PwD": 3, ...}
    Returns the closing rank for the requested category using robust matching:
      1) exact + known aliases (non-PwD first),
      2) PwD variants for the same base category,
      3) general-ish fallbacks (General),
      4) last resort: any numeric in rec (best = MIN rank).
    """
    if not isinstance(rec, dict) or not rec:
        return None

    base = _canon_cat(category) or "General"

    # Map display keys -> normalized “A-Z0-9_” for tolerant matching
    norm_map = {}
    for k, v in rec.items():
        if k is None:
            continue
        s = str(k).strip()
        if not s:
            continue
        nk = re.sub(r"[^A-Z0-9]+", "", s.upper())
        norm_map[nk] = (k, _safe_int(v))

    def _lookup(keys: list[str]) -> int | None:
        for want in keys:
            nk = re.sub(r"[^A-Z0-9]+", "", want.upper())
            if nk in norm_map:
                _, iv = norm_map[nk]
                if isinstance(iv, int):
                    return iv
        return None

    # 1) exact + aliases (non-PwD)
    alias = _cat_aliases(base)  # e.g. General -> ["General","GEN","UR","OPEN","OP","OC"]
    if base == "General":
        exact_keys = alias
    else:
        exact_keys = [base] + alias
    v = _lookup(exact_keys)
    if isinstance(v, int):
        return v

    # 2) PwD variants of the same category
    pwd_suffixes = ["_PwD", " PwD", "_PH", " PH", " PWD", "_PWD"]
    pwd_keys = []
    for a in exact_keys:
        for suf in pwd_suffixes:
            pwd_keys.append(f"{a}{suf}")
    v = _lookup(pwd_keys)
    if isinstance(v, int):
        return v

    # 3) fall back to general-ish buckets
    if base != "General":
        v = _lookup(["General", "GEN", "UR", "OPEN", "OP", "OC"])
        if isinstance(v, int):
            return v

    # 4) last resort: any numeric in rec → best (MIN rank is stronger seat)
    any_vals = [iv for (_k, iv) in norm_map.values() if isinstance(iv, int)]
    return min(any_vals) if any_vals else None

def _canon_round(v: str) -> str:
    t = str(v or "").strip().upper()
    if t in {"R1", "ROUND1", "ROUND 1", "2025_R1"}:
        return "2025_R1"
    if t in {"STRAY", "2024_STRAY", "ROUND STRAY"}:
        return "2024_Stray"
    return t or "2025_R1"


def _escape_html(s: str) -> str:
    return html.escape(str(s or ""), quote=False)

def _escape_md(s: str) -> str:
    """Very light Markdown escape, used by some formatters that keep Markdown."""
    s = str(s or "")
    return s.replace("_", r"\_").replace("*", r"\*").replace("[", r"\[") \
            .replace("]", r"\]").replace("(", r"\(").replace(")", r"\)") \
            .replace("~", r"\~").replace("`", r"\`").replace(">", r"\>") \
            .replace("#", r"\#").replace("+", r"\+").replace("-", r"\-") \
            .replace("=", r"\=").replace("|", r"\|").replace("{", r"\{") \
            .replace("}", r"\}").replace(".", r"\.")


def _load_json(path: str, fallback):
    """
    Load JSON from `path`. If file is missing or malformed, return `fallback`.
    Never throws to the caller; logs a short message instead.
    """
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.exception("Could not load JSON from %s", path)
    return fallback

def _save_json(path: str, payload) -> bool:
    """
    Save `payload` as JSON to `path`. Returns True on success, False otherwise.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        log.exception("Could not save JSON to %s", path)
        return False
def _merge_subject_map(dst: dict[str, list[dict]], data: dict):
    for subj, arr in (data or {}).items():
        if not isinstance(arr, list):
            continue
        subj_t = _norm_subject(str(subj))
        items = []
        for raw in arr:
            item = _coerce_item(raw)
            if item:
                items.append(item)
        if items:
            dst.setdefault(subj_t, []).extend(items)

def _read_array_or_questions_node(data) -> list[dict]:
    """Accept list[...] or {'questions': [...]}."""
    arr = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(arr, list):
        return []
    out = []
    for raw in arr:
        item = _coerce_item(raw)
        if item:
            out.append(item)
    return out

#----------New Quiz-------


def ensure_quiz_ready() -> None:
    """Load quiz.json (flat array) once."""
    global QUIZ_POOL, QUIZ_INDEX
    if QUIZ_POOL:
        return
    with open(QUIZ_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("quiz.json must be a flat array of question objects")
    # minimal validation
    for i, q in enumerate(data):
        if "id" not in q or "question" not in q or "options" not in q or "answer_index" not in q:
            raise ValueError(f"Bad question at index {i}: missing keys")
        if not isinstance(q["options"], list) or len(q["options"]) < 2:
            raise ValueError(f"Bad options at qid={q.get('id')}")
        if not isinstance(q["answer_index"], int) or not (0 <= q["answer_index"] < len(q["options"])):
            raise ValueError(f"Bad answer_index at qid={q.get('id')}")
    QUIZ_POOL = data
    QUIZ_INDEX = {q["id"]: q for q in QUIZ_POOL}
    log.info("✅ Loaded %d quiz items (simple loader)", len(QUIZ_POOL))

def _pick_qs(
    pool: List[Dict[str, Any]],
    *,
    subject: Optional[str] = None,
    difficulty: Optional[int] = None,
    tags_any: Optional[List[str]] = None,
    count: int = 5,
    shuffle: bool = True,
) -> List[Dict[str, Any]]:
    out = pool
    if subject:
        out = [q for q in out if q.get("subject") == subject]
    if difficulty:
        out = [q for q in out if q.get("difficulty") == difficulty]
    if tags_any:
        out = [q for q in out if any(t in (q.get("tags") or []) for t in tags_any)]
    if shuffle:
        out = list(out)
        random.shuffle(out)
    return out[:count]

def _format_question(q: Dict[str, Any], index: int, total: int) -> str:
    header_bits = [f"Q {index+1}/{total}"]
    if q.get("subject"): header_bits.append(str(q.get("subject")))
    if q.get("topic"):   header_bits.append(str(q.get("topic")))
    header = " · ".join(header_bits)
    return f"{header}\n\n{q['question']}"

def _keyboard_for(q: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(opt, callback_data=f"ans:{q['id']}:{i}")]
            for i, opt in enumerate(q["options"])]
    return InlineKeyboardMarkup(rows)

async def _send_next(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send next question or show results."""
    user_id = update.effective_user.id
    sess = QUIZ_SESSIONS.get(user_id)
    if not sess:
        await update.effective_message.reply_text("No active quiz. Use /quiz5 or /quiz10.")
        return

    i = sess["index"]
    qs: List[Dict[str, Any]] = sess["questions"]
    total = len(qs)

    if i >= total:
        # grade
        answers: Dict[str, int] = sess["answers"]
        score = 0
        parts: List[str] = []
        for q in qs:
            qid = q["id"]
            ua = answers.get(qid, -1)
            ca = q["answer_index"]
            ok = (ua == ca)
            if ok:
                score += 1
            user_txt = q["options"][ua] if 0 <= ua < len(q["options"]) else "—"
            cor_txt  = q["options"][ca]
            expl     = q.get("explanation")
            block = f"• {q['question']}\n   Your: {user_txt}\n   Correct: {cor_txt}"
            if expl:
                block += f"\n   Why: {expl}"
            parts.append(block)
        text = f"✅ Quiz complete!\nScore: {score}/{total}\n\n" + "\n\n".join(parts)
        await update.effective_message.reply_text(text)
        QUIZ_SESSIONS.pop(user_id, None)
        return

    q = qs[i]
    await update.effective_message.reply_text(
        _format_question(q, i, total),
        reply_markup=_keyboard_for(q)
    )

async def _start_quiz(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    count: int,
    subject: Optional[str] = None,
    difficulty: Optional[int] = None,
    tags_any: Optional[List[str]] = None,
) -> None:
    """Initialize a quiz session and send the first question."""
    ensure_quiz_ready()
    qs = _pick_qs(QUIZ_POOL, subject=subject, difficulty=difficulty, tags_any=tags_any, count=count, shuffle=True)
    target = update.effective_message or (update.callback_query.message if update.callback_query else None)
    if not qs:
        if target:
            await target.reply_text("No questions match those filters. Try again.")
        return
    QUIZ_SESSIONS[update.effective_user.id] = {"questions": qs, "answers": {}, "index": 0}
    await _send_next(update, context)

# public commands
async def quiz5(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _start_quiz(update, context, count=5)

async def quiz10(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _start_quiz(update, context, count=10)

async def quiz10physics(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _start_quiz(update, context, count=10, subject="Physics")

async def quiz5medium(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _start_quiz(update, context, count=5, difficulty=2)

# button click handler
async def on_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await _safe_clear_kb(query)
    data = query.data or ""
    if not data.startswith("ans:"):
        return
    try:
        _, qid, idx_str = data.split(":")
        chosen = int(idx_str)
    except Exception:
        await query.edit_message_text("Invalid answer payload.")
        return

    user_id = update.effective_user.id
    sess = QUIZ_SESSIONS.get(user_id)
    if not sess:
        await query.edit_message_text("Session expired. Use /quiz5 or /quiz10.")
        return

    qs: List[Dict[str, Any]] = sess["questions"]
    i = sess["index"]
    if i >= len(qs):
        await query.edit_message_text("Already finished. Use /quiz5 to start again.")
        return
    q = qs[i]
    if q["id"] != qid:
        # stale press
        await query.answer("That question moved on.", show_alert=False)
        return

    sess["answers"][qid] = chosen
    sess["index"] = i + 1

    # optional immediate feedback
    ca = q["answer_index"]
    user_txt = q["options"][chosen]
    cor_txt  = q["options"][ca]
    fb = "✅ Correct!" if chosen == ca else f"❌ Incorrect. Correct: {cor_txt}"
    try:
        await query.edit_message_text(f"{_format_question(q, i, len(qs))}\n\nYou picked: {user_txt}\n{fb}")
    except Exception:
        await query.message.reply_text(fb)

    await _send_next(update, context)

# small diag command
async def quizdiag(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        ensure_quiz_ready()
        await update.message.reply_text(f"Pool size: {len(QUIZ_POOL)}. First id: {QUIZ_POOL[0]['id'] if QUIZ_POOL else '—'}")
    except Exception as e:
        await update.message.reply_text(f"Quiz load error: {e}")
# ===== END SIMPLE QUIZ =====

# ===== END NEW QUIZ INTEGRATION =====

        
_CODE_COL_CANDIDATES = ["college_code", "College Code", "code", "COLLEGE_CODE"]


def _pick_name_field(rec: dict) -> str:
    for k in ("College Name", "name", "Institute", "Institute Name"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

COLLEGE_META_BY_CODE: Dict[str, dict] = {}
COLLEGE_META_BY_NAMEKEY: Dict[str, dict] = {}


def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = [str(c) for c in df.columns]
    for want in candidates:
        for c in cols:
            if c.strip().lower() == want.strip().lower():
                return c
    # also allow substring contains
    want_low = [w.lower() for w in candidates]
    for c in cols:
        lc = c.lower()
        if any(w in lc for w in want_low):
            return c
    return None

def _find_close_col(all_cols: list[str], round_tag: str) -> str | None:
    """
    Return the name of the 'closing rank' column to use for a given round tag.
    Works with many real-world headings. Example matches:
      - 'Close 2025_R1', 'Closing 2025_R1', 'Close_R1_2025'
      - '2025_R1 Close', 'Close (2025_R1)'
      - falls back to the most recent-looking 'close' column if exact round not found.
    """
    cols = [str(c) for c in all_cols]
    target = (round_tag or "").strip().lower()

    def norm(s: str) -> str:
        return s.lower().replace("(", " ").replace(")", " ").replace("_", " ").replace("-", " ").strip()

    # 1) strict: contains round_tag AND a 'close' token
    for c in cols:
        n = norm(c)
        if target and (target in n) and ("close" in n or "closing" in n):
            return c

    # 2) relaxed: contains round_tag at all
    for c in cols:
        if target and target in norm(c):
            return c

    # 3) generic fallback: pick the 'best looking' close column (latest by simple heuristics)
    closeish = [c for c in cols if "close" in norm(c) or "closing" in norm(c)]
    if not closeish:
        return None

    # prefer ones that look like they have a year or round suffix
    def score(c: str) -> tuple[int,int]:
        n = norm(c)
        has_year  = 1 if any(y in n for y in ("2025","2024","2023","2022")) else 0
        has_rtag  = 1 if any(rt in n for rt in ("r1","r2","r3","r4")) else 0
        return (has_year+has_rtag, len(n))

    closeish.sort(key=score, reverse=True)
    return closeish[0]


def _find_sheet_like(xl: "pd.ExcelFile", preferred: str = "Colleges") -> Optional[str]:
    # exact
    for s in xl.sheet_names:
        if s.strip().lower() == preferred.lower():
            return s
    # fuzzy
    for s in xl.sheet_names:
        if "college" in s.strip().lower():
            return s
    # fallback first sheet
    return xl.sheet_names[0] if xl.sheet_names else None

# ========================= Colleges dataset + name index =========================
COLLEGE_NAME_BY_KEY: dict[str, str] = {}   # e.g. "C0007" -> "All India Institute of Medical Sciences, New Delhi"
COLLEGE_META_INDEX: dict[str, dict] = {}   # canonical lookup for metadata by key (code or normalized name)
COLLEGE_NAME_BY_CODE: dict[str, str] = {}
COLLEGE_NAME_BY_ID:   dict[str, str] = {}
COLLEGE_NIRF_BY_CODE: dict[str, int] = {}
COLLEGE_FEE_BY_CODE:  dict[str, int] = {}


def _pick(df: pd.DataFrame, *cands: str) -> Optional[str]:
    """Pick the first matching column name from candidates (case-insensitive, forgiving)."""
    if df is None or df.empty:
        return None
    cols = [str(c) for c in df.columns]
    norm = {c.lower().strip(): c for c in cols}
    for want in cands:
        key = want.lower().strip()
        if key in norm:
            return norm[key]
        # contains-style fallback
        for k, orig in norm.items():
            if key == k or key in k:
                return orig
    return None

def build_code_to_name_index(path: str) -> dict[str, str]:
    """
    Reads the 'Colleges' (or nearest) sheet and returns a dict: College Code -> College Name (display string).
    Also updates COLLEGE_NAME_BY_KEY global for fast access elsewhere (e.g., shortlist display).
    """
    global COLLEGE_NAME_BY_KEY, COLLEGE_META_INDEX
    COLLEGE_NAME_BY_KEY = {}
    

    try:
        xl = pd.ExcelFile(path)
    except Exception as e:
        log.exception("build_code_to_name_index: could not open Excel: %s", path)
        return {}

    # Find a suitable sheet for colleges metadata
    sheet = None
    for s in xl.sheet_names:
        if s.strip().lower() in {"colleges", "college", "institutes", "metadata"}:
            sheet = s
            break
    if not sheet:
        # fallback: first sheet that has likely columns
        for s in xl.sheet_names:
            try:
                probe = pd.read_excel(xl, sheet_name=s, nrows=3)
            except Exception:
                continue
            if _pick(probe, "College Name", "Institute Name") and _pick(probe, "College Code", "Code", "ID", "College ID"):
                sheet = s
                break

    if not sheet:
        log.warning("build_code_to_name_index: no 'Colleges' sheet found; name index will be empty.")
        return {}

    try:
        df = pd.read_excel(xl, sheet_name=sheet).dropna(how="all")
    except Exception as e:
        log.exception("build_code_to_name_index: read failed for sheet '%s'", sheet)
        return {}

    # Column picks
    col_code = _pick(df, "College Code", "Code", "College_ID", "College Id", "ID", "CollegeID")
    col_name = _pick(df, "College Name", "Institute Name", "College", "Institute")
    col_city = _pick(df, "City", "Location")
    col_state = _pick(df, "State")
    col_site = _pick(df, "Website", "URL", "Institute URL")
    col_nirf = _pick(df, "NIRF Rank", "NIRF")
    col_fee = _pick(df, "Total Fee", "Total Fees", "Fee", "Fees", "Tuition Fee")

    if not col_name:
        log.warning("build_code_to_name_index: no College Name column in '%s'", sheet)

    # Build maps
    for _, r in df.iterrows():
        name = str(r.get(col_name) or "").strip()
        code = str(r.get(col_code) or "").strip().upper()
        if not name:
            continue

        # Prefer a code key if available, else fall back to normalized name-key
        key = code if code else _name_key(name)

        COLLEGE_NAME_BY_KEY[key] = name
        COLLEGE_META_INDEX[key] = {
            "college_name": name,
            "college_code": code or None,
            "city": (str(r.get(col_city)) if col_city else "") or "",
            "state": (str(r.get(col_state)) if col_state else "") or "",
            "website": (str(r.get(col_site)) if col_site else "") or "",
            "nirf_rank": _safe_int(r.get(col_nirf)) if col_nirf else None,
            "total_fee": _safe_int(r.get(col_fee)) if col_fee else None,
        }

    log.info("Loaded %d college metadata rows from sheet '%s'", len(COLLEGE_NAME_BY_KEY), sheet)
    return COLLEGE_NAME_BY_KEY


def load_colleges_dataset(
    xlsx_path: str,
    meta_sheet: str = "Colleges",
    cutoff_sheet: str = "Cutoffs",
) -> pd.DataFrame:
    """
    Robust loader for colleges metadata.
    1) Try reading 'Colleges' sheet.
    2) If empty or missing, build a minimal DF from distinct codes/ids in 'Cutoffs'.
    Returns a DataFrame (possibly minimal) — never None.
    """
    log = logging.getLogger("aceit-bot")

    try:
        xl = pd.ExcelFile(xlsx_path)
    except Exception as e:
        log.exception("Failed to open Excel '%s': %s", xlsx_path, e)
        return pd.DataFrame()

    # Helper to pick columns forgivingly
    def _pick(cols, *cands):
        cols = [str(c) for c in cols]
        norm = {c.lower().strip(): c for c in cols}
        for want in cands:
            w = want.lower().strip()
            if w in norm:
                return norm[w]
            for k, orig in norm.items():
                if w == k or w in k:
                    return orig
        return None

    # 1) Try full Colleges sheet
    if meta_sheet in xl.sheet_names:
        try:
            df = pd.read_excel(xl, sheet_name=meta_sheet)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # normalize a few canonical columns
                cols = list(df.columns)
                c_name = _pick(cols, "College Name", "Institute Name", "Name")
                c_code = _pick(cols, "college_code", "College Code", "Code")
                c_id   = _pick(cols, "college_id", "College ID", "ID")
                c_state= _pick(cols, "State")
                c_nirf = _pick(cols, "NIRF", "NIRF Rank")
                c_fee  = _pick(cols, "Fee", "Total Fee")
                c_site = _pick(cols, "Website")

                out = pd.DataFrame()
                if c_name: out["College Name"] = df[c_name].astype(str).str.strip()
                if c_code: out["college_code"] = df[c_code].astype(str).str.strip()
                if c_id:   out["college_id"]   = df[c_id].astype(str).str.strip()
                if c_state:out["State"]        = df[c_state].astype(str).str.strip()
                if c_nirf: out["NIRF"]         = df[c_nirf]
                if c_fee:  out["Fee"]          = df[c_fee]
                if c_site: out["Website"]      = df[c_site].astype(str).str.strip()

                # keep all original columns too (don’t destructively lose data)
                for c in df.columns:
                    if c not in out.columns:
                        out[c] = df[c]

                log.info("Loaded %d college metadata rows from sheet '%s'", len(out), meta_sheet)
                return out
            else:
                log.warning("Sheet '%s' is empty in '%s'", meta_sheet, xlsx_path)
        except Exception:
            log.exception("Failed to read sheet '%s' in '%s'", meta_sheet, xlsx_path)
    else:
        log.warning("Sheet '%s' not found in '%s'", meta_sheet, xlsx_path)

    # 2) Fallback: synthesize minimal DF from Cutoffs distinct codes/ids
    if cutoff_sheet in xl.sheet_names:
        try:
            cuts = pd.read_excel(xl, sheet_name=cutoff_sheet)
            if cuts is None or cuts.empty:
                log.warning("Cutoffs sheet '%s' exists but is empty; returning empty colleges DF", cutoff_sheet)
                return pd.DataFrame()

            c_code = _pick(cuts.columns, "college_code", "College Code", "Code")
            c_id   = _pick(cuts.columns, "college_id",   "College ID",   "ID")
            c_name = _pick(cuts.columns, "College Name", "Institute Name", "Name")  # some cutoffs keep names, some don't
            c_state= _pick(cuts.columns, "State")  # sometimes present

            rows = []
            # Prefer code; if missing use id; last resort name-only row
            if c_code and cuts[c_code].notna().any():
                for code in sorted({str(v).strip() for v in cuts[c_code].dropna().unique()}):
                    entry = {"college_code": code}
                    if c_id:
                        # bring a matching id if there is an obvious 1:1 present in the sheet
                        id_candidates = cuts[cuts[c_code].astype(str).str.strip() == code]
                        if c_id in id_candidates and id_candidates[c_id].notna().any():
                            entry["college_id"] = str(id_candidates[c_id].dropna().iloc[0]).strip()
                    if c_name:
                        name_candidates = cuts[cuts[c_code].astype(str).str.strip() == code]
                        if c_name in name_candidates and name_candidates[c_name].notna().any():
                            entry["College Name"] = str(name_candidates[c_name].dropna().iloc[0]).strip()
                    if c_state:
                        st_candidates = cuts[cuts[c_code].astype(str).str.strip() == code]
                        if c_state in st_candidates and st_candidates[c_state].notna().any():
                            entry["State"] = str(st_candidates[c_state].dropna().iloc[0]).strip()
                    rows.append(entry)
            elif c_id and cuts[c_id].notna().any():
                for cid in sorted({str(v).strip() for v in cuts[c_id].dropna().unique()}):
                    entry = {"college_id": cid}
                    if c_name:
                        name_candidates = cuts[cuts[c_id].astype(str).str.strip() == cid]
                        if c_name in name_candidates and name_candidates[c_name].notna().any():
                            entry["College Name"] = str(name_candidates[c_name].dropna().iloc[0]).strip()
                    if c_state:
                        st_candidates = cuts[cuts[c_id].astype(str).str.strip() == cid]
                        if c_state in st_candidates and st_candidates[c_state].notna().any():
                            entry["State"] = str(st_candidates[c_state].dropna().iloc[0]).strip()
                    rows.append(entry)
            elif c_name and cuts[c_name].notna().any():
                # absolute last resort: names only
                for nm in sorted({str(v).strip() for v in cuts[c_name].dropna().unique()}):
                    rows.append({"College Name": nm})

            out = pd.DataFrame(rows)
            log.warning(
                "Colleges sheet missing/empty; synthesized minimal colleges DF from cutoffs: %d rows",
                len(out),
            )
            return out
        except Exception:
            log.exception("Failed to synthesize colleges DF from cutoffs sheet '%s'", cutoff_sheet)

    # 3) Nothing worked
    log.warning("Returning empty colleges DataFrame")
    return pd.DataFrame()


# ---------- Flexible column finder ----------
def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first column whose normalized name contains any candidate token."""
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


def build_name_maps_from_colleges_df(df):
    """Populate COLLEGE_NAME_BY_CODE / COLLEGE_NAME_BY_ID / COLLEGE_META_INDEX from the Colleges DF."""
    global COLLEGE_NAME_BY_CODE, COLLEGE_NAME_BY_ID, COLLEGE_META_INDEX

    if df is None or len(df) == 0:
        COLLEGE_NAME_BY_CODE = {}
        COLLEGE_NAME_BY_ID   = {}
        COLLEGE_META_INDEX   = {}
        return

    cols = list(df.columns)
    c_name = _pick_col(cols, "College Name", "Institute Name", "Name") or cols[0]
    c_code = _pick_col(cols, "college_code", "College Code", "Code")
    c_id   = _pick_col(cols, "college_id",   "College ID",   "ID")

    by_code, by_id, meta = {}, {}, {}
    for _, r in df.iterrows():
        nm   = (str(r.get(c_name)) or "").strip()
        code = _norm_key(r.get(c_code)) if c_code else ""
        cid  = _norm_key(r.get(c_id))   if c_id   else ""

        if code and nm:
            by_code[code] = nm
        if cid and nm:
            by_id[cid] = nm

        # also keep a meta index (by code/id/namekey for possible future use)
        if code:
            meta[code] = dict(r)
        if cid:
            meta[cid] = dict(r)
        if nm:
            meta[_name_key(nm)] = dict(r)

    COLLEGE_NAME_BY_CODE = by_code
    COLLEGE_NAME_BY_ID   = by_id
    COLLEGE_META_INDEX   = meta

    import logging as _lg
    _lg.getLogger("aceit-bot").info(
        "Name maps primed: codes=%d ids=%d meta_keys=%d",
        len(COLLEGE_NAME_BY_CODE), len(COLLEGE_NAME_BY_ID), len(COLLEGE_META_INDEX)
    )

# ---------- Cutoff lookup loader (complete) ----------

def load_cutoff_lookup_from_excel(
    path: str,
    sheet: str,
    *,
    round_tag: str,          # "2025_R1" or "2024_Stray"
    require_quota: str = "AIQ",
    require_course_contains: str = "MBBS",
    require_category_set = ("General","EWS","OBC","SC","ST"),
) -> dict[tuple[str, str, str], int]:
    """
    Strict loader: returns {(key, quota, category): closing_rank}
    key = normalized college_code OR college_id (name-only is NOT allowed)
    Only loads rows that can be proven to be MBBS + required quota + category + round/year.
    If constraints cannot be proven, returns {} (fail-closed).
    """
    import pandas as pd, re

    def _pick(cols, *cands):
        cols = [str(c) for c in cols]
        low  = {c.lower().strip(): c for c in cols}
        for w in cands:
            ww = w.lower().strip()
            if ww in low:
                return low[ww]
            for k, orig in low.items():
                if ww in k:
                    return orig
        return None

    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except Exception as e:
        log.exception("[cutoffs] open failed: %s", e)
        return {}

    if df is None or df.empty:
        log.warning("[cutoffs] empty sheet: %s", sheet)
        return {}

    cols = [str(c) for c in df.columns]
    c_code   = _pick(cols, "college_code", "College Code", "Code", "institute_code")
    c_id     = _pick(cols, "college_id", "College ID", "ID")
    c_name   = _pick(cols, "College Name", "college_name", "Institute Name")  # not used for keys
    c_quota  = _pick(cols, "Quota", "Seat Type", "Allotment", "Allotment Category")
    c_cat    = _pick(cols, "Category", "Seat Category", "Cat")
    c_close  = _pick(cols, "Closing Rank", "ClosingRank", "Close Rank", "Closing AIR")
    c_course = _pick(cols, "Course")
    c_round  = _pick(cols, "Round", "Round Tag", "Round Name", "Round No")
    c_year   = _pick(cols, "Year", "as_of_year")
    c_pwd    = _pick(cols, "PwD (Y/N)", "PwD", "PwBD", "PH")

    # Must have these structural columns
    if not c_close or not c_quota or not c_cat or not c_course:
        log.warning("[cutoffs] strict: missing mandatory columns (close/quota/cat/course)")
        return {}

    # --- Canonical quota + filter (AFTER detecting the correct quota column) ---
    df[c_quota] = df[c_quota].astype(str)
    df["_quota"] = df[c_quota].apply(_canon_quota)

    if require_quota and str(require_quota).strip():
        want_quota = _canon_quota(require_quota)
        before = len(df)
        df = df[df["_quota"] == want_quota]
        log.info("[cutoffs] quota filter kept %d/%d rows (bucket=%s)", len(df), before, want_quota)

    if df.empty:
        return {}

    # --- Filter course (e.g., MBBS) ---
    df = df[df[c_course].astype(str).str.contains(require_course_contains, case=False, na=False)]
    if df.empty:
        return {}

    # --- Derive year/round from round_tag and filter if those cols exist ---
    yr = None
    m = re.search(r"(20\d{2})", str(round_tag))
    yr = int(m.group(1)) if m else None

    rnum = None
    m2 = re.search(r"R\s*([1-4])", str(round_tag).upper())
    rnum = int(m2.group(1)) if m2 else None

    if c_year and yr:
        df = df[df[c_year].astype(str).str.contains(str(yr), na=False)]
        if df.empty:
            return {}

    if c_round and rnum is not None:
        mask = df[c_round].astype(str).str.contains(fr"\b{rnum}\b", case=False, na=False) | \
               df[c_round].astype(str).str.upper().str.contains(f"R{rnum}", na=False)
        df = df[mask]
        if df.empty:
            return {}

    # --- Canonical category + filter set ---
    df["_cat_raw"] = df[c_cat].astype(str)
    df["_cat"] = df[c_cat].map(_canon_cat)
    df = df[df["_cat"].isin(list(require_category_set))]
    if df.empty:
        return {}

    # --- Exclude PwD when main category isn’t *_PwD ---
    if c_pwd:
        # treat N/NO/0/"" as non-PwD
        non_pwd_mask = df[c_pwd].astype(str).str.strip().str.upper().isin(["N", "NO", "0", ""])
        want_non_pwd = ~df["_cat"].str.endswith("_PwD", na=False)
        df = df[(want_non_pwd & non_pwd_mask) | (~want_non_pwd)]
        if df.empty:
            return {}

    # --- Closing rank numeric ---
    df["_close"] = pd.to_numeric(df[c_close], errors="coerce").astype("Int64")
    df = df.dropna(subset=["_close"])
    if df.empty:
        return {}

    # --- Build strict lookup (prefer the loosest closing rank if duplicates appear) ---
    out: dict[tuple[str,str,str], int] = {}
    used = 0

    for _, r in df.iterrows():
        k_code = _norm_key(r.get(c_code)) if c_code else ""
        k_id   = _norm_key(r.get(c_id))   if c_id   else ""
        key = k_code or k_id
        if not key:
            continue  # STRICT: require code or id; no name-only keys

        q = str(r["_quota"])
        c = str(r["_cat"])
        v = int(r["_close"])
        cur = out.get((key, q, c))
        if cur is None or v > cur:
            out[(key, q, c)] = v
            used += 1

    log.info("[cutoffs] STRICT loaded: %d triplets (sheet='%s', round=%s quota=%s course~%s)",
             len(out), sheet, round_tag, require_quota, require_course_contains)
    return out

    
# ---------- Globals used later elsewhere ----------
CUTOFFS: Dict[str, Dict[str, Dict[str, int]]] = {}
CUTOFFS_Q: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {"2025_R1": {}, "2024_Stray": {}}

   

COLLEGES: List[Dict[str, Any]] = []
COLLEGE_META_INDEX: Dict[str, Dict[str, Any]] = {}
MODE_KEY = "active_flow"

DISPLAY_NAME_KEYS  = ["College Name", "College", "Institute", "Institute Name", "Name"]
DISPLAY_STATE_KEYS = ["State", "STATE", "State (Normalized)"]
DISPLAY_NAME_COLS  = ["College Name", "College", "Institute", "Institute Name"]

# ========================= Flow lock (prevent mixing quiz/predict/ask) =========================

def _ensure_flow_or_bounce(update: Update, context: ContextTypes.DEFAULT_TYPE, want: str) -> bool:
    """
    Synchronous guard: ensure only one flow at a time.
    If another flow is active, politely bounce and return False.
    Otherwise, mark `want` as active and return True.
    """
    cur = context.user_data.get(MODE_KEY)
    if cur and cur != want:
        # pick the right target (message vs callback)
        tgt = _target(update)
        if tgt:
            # We can't `await` here (this helper is sync), so schedule the send
            try:
                context.application.create_task(
                    tgt.reply_text(
                        f"You're currently in *{cur}*. Send /cancel to exit before starting *{want}*.",
                        parse_mode="Markdown",
                    )
                )
            except Exception:
                pass
        return False

    # claim the lock
    context.user_data[MODE_KEY] = want
    return True


def _start_flow(context: ContextTypes.DEFAULT_TYPE, name: str) -> Optional[str]:
    """
    If another flow is active, return its name so the caller can tell the user to /cancel.
    Otherwise, mark this flow as active and return None.
    """
    cur = context.user_data.get(MODE_KEY)
    if cur and cur != name:
        return cur
    context.user_data[MODE_KEY] = name
    return None

def _end_flow(context: ContextTypes.DEFAULT_TYPE, name: str) -> None:
    """Clear the lock if it matches the given name."""
    if context.user_data.get(MODE_KEY) == name:
        context.user_data.pop(MODE_KEY, None)

def unlock_flow(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Force-clear any active flow lock (used on /start, /menu, /cancel)."""
    context.user_data.pop(MODE_KEY, None)


def _target(update: Update):
    """
    Return the correct message object to reply to, whether the update
    came from a normal message or a callback button press.
    """
    if update is None:
        return None
    if getattr(update, "message", None):
        return update.message
    cq = getattr(update, "callback_query", None)
    if cq and getattr(cq, "message", None):
        return cq.message
    return None



DEFAULT_CHUNK = 3600
async def _send_chunked_results_plain(
    context,
    chat_id: int,
    header_text: str,
    rows_text: list[str],
    *,
    max_items_per_msg: int = 15,
) -> None:
    """
    Sends the header once, then batches rows so each message stays < TG_LIMIT chars.
    No Markdown parse mode to avoid formatting surprises.
    """
    # Fallback if someone forgot to define TG_LIMIT
    limit = int(globals().get("TG_LIMIT", 4000))

    # If there are no rows, just send header.
    if not rows_text:
        await context.bot.send_message(chat_id=chat_id, text=header_text)
        return

    first = True
    i = 0
    n = len(rows_text)
    while i < n:
        # start with a logical batch size
        batch = rows_text[i : i + max_items_per_msg]
        body  = "\n\n".join(batch)
        text  = (header_text + "\n\n" + body) if first else body

        # If text is too long, shrink the batch until it fits.
        # (Telegram hard cap is ~4096; we use TG_LIMIT.)
        while len(text) > limit and len(batch) > 1:
            batch = batch[:-1]
            body  = "\n\n".join(batch)
            text  = (header_text + "\n\n" + body) if first else body

        # If even one row is gigantic (shouldn’t happen), truncate it hard.
        if len(text) > limit and len(batch) == 1:
            text = text[: limit - 20] + "…"

        await context.bot.send_message(chat_id=chat_id, text=text)

        i += len(batch)
        first = False
        header_text = "Continued…"
        
TG_LIMIT = int(globals().get("TG_LIMIT", 4000))


async def coach_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Notes-only coach: uses last_predict_shortlist, keeps order, no crosses."""
    tgt = _target(update) or update.effective_message
    ud = context.user_data or {}
    shortlist = ud.get("last_predict_shortlist") or []
    air = ud.get("last_predict_air")

    if not shortlist:
        await tgt.reply_text("Run /predict first, then tap 🧠 AI Notes.")
        return ConversationHandler.END

    # top 10, keep order
    facts = []
    for i, r in enumerate(shortlist[:10], 1):
        facts.append({
            "rank": i,
            "code": r.get("college_code"),
            "name": r.get("college_name"),
            "state": r.get("state"),
            "closing_rank": _to_int(r.get("ClosingRank")),
            "nirf": _to_int(r.get("nirf_rank_medical_latest")),
            "fee": _to_fee_lakh(r.get("total_fee")),
            "ownership": r.get("ownership"),
            "hostel": (True if r.get("hostel_available") is True else None),
        })

    text = _notes_via_llm(facts, air) or _notes_deterministic(facts, air)

    await tgt.reply_text(text, parse_mode="Markdown", disable_web_page_preview=True)
    return ConversationHandler.END

async def coach_adjust_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = update.effective_user.id if update.effective_user else "anon"
    cache = context.application.bot_data.get("AI_COACH_CACHE", {}).get(uid)
    if not cache:
        await q.edit_message_text("Session expired. Send /coach again.")
        return

    prefs = cache["prefs"]
    mode = (q.data or "").split(":", 1)[-1]
    if mode == "safer":
        prefs["target_mix"] = {"safe": 60, "moderate": 30, "dream": 10}
    elif mode == "dreamier":
        prefs["target_mix"] = {"safe": 20, "moderate": 40, "dream": 40}
    else:
        prefs["target_mix"] = {"safe": 40, "moderate": 40, "dream": 20}

    cands = cache["candidates"]
    try:
        plan = _call_ai_coach(cands, prefs)
    except Exception:
        # fallback: just re-order deterministically
        df = pd.DataFrame(cands)
        df["ClosingRank"] = df["ClosingRank"].fillna(999999)
        plan = _fallback_rank(df, int(context.user_data.get("rank_air", 999999)))

    cache["plan"] = plan.model_dump()
    cache["prefs"] = prefs

    by_code = {str(x["college_code"]): x for x in cands}
    lines = [f"*Updated Mix:* {prefs['target_mix']['safe']}% safe · {prefs['target_mix']['moderate']}% moderate · {prefs['target_mix']['dream']}% dream\n"]
    n=1
    for code in plan.ordered_codes[:COACH_SHOW_N]:
        row = by_code.get(code, {})
        nm = row.get("college_name","?")
        rz = plan.rationales.get(code) or ""
        lines.append(f"*{n}. {nm}* (`{code}`) — { _trim(rz) }")
        n += 1

    await q.edit_message_text("\n".join(lines), parse_mode="Markdown",
                              reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("💾 Save as My List", callback_data=f"{COACH_SAVE}:v1")],
                                [
                                    InlineKeyboardButton("Make Safer", callback_data=f"{COACH_ADJUST}:safer"),
                                    InlineKeyboardButton("Balanced",   callback_data=f"{COACH_ADJUST}:balanced"),
                                    InlineKeyboardButton("More Dream", callback_data=f"{COACH_ADJUST}:dreamier"),
                                ],
                              ]))

async def coach_save_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = update.effective_user.id if update.effective_user else "anon"
    cache = context.application.bot_data.get("AI_COACH_CACHE", {}).get(uid)
    if not cache:
        await q.edit_message_text("Nothing to save. Run /coach first.")
        return

    plan = CoachPlan(**cache["plan"])
    mylists = context.application.bot_data.setdefault("MYLISTS", {})
    lists_for_user = mylists.setdefault(uid, [])
    version = len(lists_for_user) + 1
    lists_for_user.append({
        "version": version,
        "ts": time.time(),
        "ordered_codes": plan.ordered_codes,
        "risk_mix": plan.risk_mix,
        "rationales": plan.rationales
    })
    await q.edit_message_text(f"✅ Saved as *My List v{version}*. Use /mylist to view.", parse_mode="Markdown")

async def send_long_message(bot, chat_id: int, text: str, parse_mode: str | None = None):
    """
    Safe long-message sender for Telegram. Splits on blank lines first,
    then hard-slices if absolutely necessary. Mirrors the behavior of
    _send_chunked_results_plain but for a single long text block.
    """
    if not text:
        return

    limit = TG_LIMIT
    if len(text) <= limit:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
        return

    # Try to split on paragraph boundaries to preserve formatting
    parts = text.split("\n\n")
    batch = []
    cur = ""
    sent_any = False

    def _sendable(prefix: bool, s: str) -> str:
        return ("Continued…\n\n" + s) if prefix else s

    for p in parts:
        candidate = (cur + ("\n\n" if cur else "") + p)
        if len(_sendable(sent_any, candidate)) <= limit:
            cur = candidate
        else:
            # flush current
            if cur:
                await bot.send_message(chat_id=chat_id, text=_sendable(sent_any, cur), parse_mode=parse_mode)
                sent_any = True
            # if a single paragraph is too big, hard-slice it
            if len(_sendable(sent_any, p)) > limit:
                chunk = p
                while len(_sendable(sent_any, chunk)) > limit:
                    # reserve a few chars for ellipsis
                    cut = limit - (15 if sent_any else 15)
                    await bot.send_message(chat_id=chat_id,
                                           text=_sendable(sent_any, chunk[:cut] + "…"),
                                           parse_mode=parse_mode)
                    sent_any = True
                    chunk = chunk[cut:]
                cur = chunk
            else:
                cur = p

    if cur:
        await bot.send_message(chat_id=chat_id, text=_sendable(sent_any, cur), parse_mode=parse_mode)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Choose an option:",
        reply_markup=main_menu_keyboard()
    )

# ========================= Excel helpers & loaders =========================
def _norm_hdr(s: str) -> str:
    s = unidecode(str(s or ""))
    s = s.replace("’", "'").replace("“", "\"").replace("”", "\"")
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()

def _tok_hdr(h: str) -> List[str]:
    return [t for t in re.split(r"[^A-Z0-9]+", str(h).upper()) if t]

def _clean_int(x) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    s = re.sub(r"[,\s]", "", s)
    try:
        return int(float(s))
    except Exception:
        m = re.search(r"\d+", s)
        return int(m.group(0)) if m else None


def _find_header_row(df_raw: pd.DataFrame) -> int:
    """Guess the header row by scanning early rows for signals of name+category columns."""
    max_scan = min(40, len(df_raw))
    cat_tokens = {"GENERAL", "UR", "GEN", "OPEN", "EWS", "OBC", "BC", "SC", "ST", "QUOTA"}
    for i in range(max_scan):
        row = [str(x) for x in df_raw.iloc[i].tolist()]
        normed = [_norm_hdr(" ".join(r.split())) for r in row]
        has_name = any(("COLLEGE" in n) or ("INSTITUTE" in n) for n in normed)
        has_any_cat = any(n in cat_tokens or any(tok in n for tok in cat_tokens) for n in normed)
        if has_name and has_any_cat:
            return i
    return 0

def _is_number(x):
    try:
        if x is None: return False
        s = str(x).strip().replace(",", "")
        float(s)
        return True
    except Exception:
        return False

def _is_non_pwd(v) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in {"", "n", "no", "0", "false"}

def _is_mbbs(v) -> bool:
    return "MBBS" in str(v or "").upper()

def _round_tokens(rk: str) -> List[str]:
    if rk == "2025_R1":      return ["R1", "ROUND 1", " 1 "]
    if rk == "2024_Stray":   return ["STRAY"]
    return []

def _strict_closing_col(df: pd.DataFrame) -> Optional[str]:
    """Try hard to locate a 'closing rank' like column."""
    cols = [str(c) for c in df.columns]
    candidates = []
    for c in cols:
        h = _norm_hdr(c)
        if ("CLOS" in h or "CUTOFF" in h) and ("RANK" in h or "AIR" in h):
            candidates.append(c)
    if not candidates:
        for c in cols:
            h = _norm_hdr(c)
            if ("AIR" in h) and not any(bad in h for bad in ("NIRF", "YEAR", "ROUND", "FEE", "PIN", "CODE", "SEAT", "INTAKE", "CAPACITY")):
                candidates.append(c)

    best = None
    best_score = -10**9
    for c in candidates:
        ser = pd.to_numeric(df[c], errors="coerce").dropna()
        if ser.empty:
            continue
        med = float(ser.median())
        mx = float(ser.max())
        score = 0
        if med >= 500:          score += 5
        if mx >= 1000:          score += 6
        if mx <= 3_000_000:     score += 2
        if 100 <= med <= 400_000: score += 3
        h = _norm_hdr(c)
        if any(bad in h for bad in ("NIRF", "YEAR", "ROUND", "FEE", "PIN", "CODE", "SEAT", "INTAKE", "CAPACITY")):
            score -= 80
        if score > best_score:
            best_score, best = score, c
    return best

def _series_is_ranklike(series: pd.Series) -> bool:
    ser = pd.to_numeric(series, errors="coerce").dropna()
    if ser.empty:
        return False
    med = float(ser.median())
    p90 = float(ser.quantile(0.9))
    mx = float(ser.max())
    # Seat counts are small; ranks have larger spread.
    return (med >= 200) or (p90 >= 1000) or (mx >= 1000)

def choose_cutoff_sheet(xl: pd.ExcelFile, round_key: str) -> Optional[str]:
    """Heuristically choose the best cutoffs sheet for a given round_key."""
    ov = (CUTSHEET_OVERRIDE or {}).get(round_key)
    if ov and ov in xl.sheet_names:
        return ov

    def _wide_metrics(df: pd.DataFrame) -> tuple[int, bool]:
        cols = [str(c) for c in df.columns]
        name_col = _pick_col(cols, "College Name", "Institute Name", "College", "Institute")
        quota_col = _pick_col(cols, "Quota", "Allotment", "Allotment Category", "Seat Type", "seat_type")
        if not (name_col and quota_col):
            return (0, False)
        cat_cols = []
        probes = ["General", "Open", "UR", "GEN", "OP", "OPN",
                  "Ews", "EWS", "Obc", "OBC", "BC", "Sc", "SC", "St", "ST"]
        for p in probes:
            c = _pick_col(cols, p)
            if c:
                nc = _norm_hdr(c)
                if ("PWD" in nc) or ("PH" in nc) or any(b in nc for b in ("SEAT", "INTAKE", "CAPACITY", "TOTAL")):
                    continue
                if _series_is_ranklike(df[c]):
                    cat_cols.append(c)
        cat_cols = list(dict.fromkeys(cat_cols))
        if len(cat_cols) < 2:
            return (0, False)
        sample = df.head(300)
        hits = 0
        for _, r in sample.iterrows():
            for c in cat_cols:
                v = pd.to_numeric(pd.Series([r.get(c)]), errors="coerce").iloc[0]
                if pd.notna(v):
                    iv = int(v)
                    if 1 <= iv <= 3_000_000:
                        hits += 1
        return (hits, True)

    def _long_metrics(df: pd.DataFrame) -> tuple[int, bool]:
        cols = [str(c) for c in df.columns]
        name_col = _pick_col(cols, "College Name", "Institute Name", "College", "Institute")
        cat_col = _pick_col(cols, "Category", "Seat Category", "Cat")
        close_col = _pick_col(cols, "ClosingRank", "Closing Rank", "Close Rank", "AIR", "Closing AIR", "Closing")
        if close_col is None:
            close_col = _strict_closing_col(df)
        if not (name_col and cat_col and close_col):
            return (0, False)
        sample = df.head(400).copy()
        sample["_cat"] = sample[cat_col].apply(_canon_cat_from_value)
        sample["_rank"] = pd.to_numeric(sample[close_col], errors="coerce")
        valid = sample.dropna(subset=["_cat", "_rank"])
        return (len(valid), True)

    best_name = None
    best_score = -10**9
    for s in xl.sheet_names:
        sn = _norm_hdr(s)
        if any(bad in sn for bad in ["CODE", "MAPPING", "README", "LEGEND", "COLLEGES", "CATEGORY CODE"]):
            continue
        try:
            df0 = pd.read_excel(xl, sheet_name=s, header=None, nrows=60)
            hdr = _find_header_row(df0)
            df = pd.read_excel(xl, sheet_name=s, header=hdr, nrows=160)
        except Exception:
            continue

        wide_hits, is_wide = _wide_metrics(df)
        long_hits, is_long = _long_metrics(df)

        score = 0
        score += 5 * wide_hits
        score += 2 * long_hits
        if is_wide:
            score += 200
        if is_long and long_hits > 0:
            score += 40
        if round_key == "2025_R1":
            if "2025" in sn: score += 6
            if "R1" in sn or "ROUND 1" in sn: score += 6
        elif round_key == "2024_Stray":
            if "2024" in sn: score += 6
            if "STRAY" in sn: score += 6

        if score > best_score:
            best_score = score
            best_name = s

    return best_name

def load_college_metadata_from_excel(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    try:
        xl = pd.ExcelFile(path)
    except Exception:
        log.exception("Could not open Excel for metadata")
        return {}

    # find a likely "Colleges" sheet
    sheet = None
    for s in xl.sheet_names:
        if _norm_hdr(s) == "COLLEGES":
            sheet = s
            break
    if not sheet:
        for s in xl.sheet_names:
            try:
                dfp = pd.read_excel(path, sheet_name=s, nrows=5)
            except Exception:
                continue
            if any(_norm_hdr(c) in {"COLLEGE NAME", "INSTITUTE NAME", "COLLEGE", "INSTITUTE"} for c in dfp.columns):
                sheet = s
                break
    if not sheet:
        log.info("No 'colleges' sheet found; metadata fallback will be minimal.")
        return {}

    df = pd.read_excel(path, sheet_name=sheet).dropna(how="all")
    cols = [str(c) for c in df.columns]

    name_col = _pick_col(cols, "College Name", "Institute Name", "College", "Institute")
    state_col = _pick_col(cols, "State")
    city_col  = _pick_col(cols, "City", "Location")
    web_col   = _pick_col(cols, "Website", "URL", "Institute URL")
    nirf_col  = _pick_col(cols, "NIRF Rank", "NIRF")
    fee_col   = _pick_col(cols, "Total Fee", "Total Fees", "Fee", "Fees", "Tuition Fee")

    # NEW: try to pick the code column used in Cutoffs (e.g., "Code", "College Code", etc.)
    code_col  = _pick_col(cols, "Code", "College Code", "Institute Code", "MCC Code", "CCode", "CLG Code")

    if not name_col:
        return {}

    out: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        nm = _clean_str(row.get(name_col))
        if not nm:
            continue
        key = re.sub(r"[^A-Z0-9]+", "", nm.upper())
        out[key] = {
            "name": nm,
            "state": _clean_str(row.get(state_col)) if state_col else "",
            "city": _clean_str(row.get(city_col)) if city_col else "",
            "website": _clean_str(row.get(web_col)) if web_col else "",
            "nirf_rank": _clean_int(row.get(nirf_col)) if nirf_col else None,
            "total_fees": _clean_int(row.get(fee_col)) if fee_col else None,
            "policy": {"pg_quota_available": False, "bond_service": {"has_bond": False}},
            # store the raw code if present
            "code": _clean_str(row.get(code_col)) if code_col else "",
        }
    log.info("Loaded %d college metadata rows from sheet '%s'", len(out), sheet)
    return out


# ========================= Profiles (persisted JSON) =========================
PROFILES_PATH = "profiles.json"

def _load_profiles() -> Dict[str, Any]:
    try:
        if os.path.exists(PROFILES_PATH):
            with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        log.exception("Failed to load profiles.json")
    return {}

def _save_profiles(store: Dict[str, Any]) -> None:
    try:
        with open(PROFILES_PATH, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
    except Exception:
        log.exception("Failed to save profiles.json")

PROFILES = _load_profiles()

def _uid(update: Update) -> str:
    user = update.effective_user
    return str(user.id) if user else "unknown"

def get_user_profile(update: Update) -> Dict[str, Any]:
    return PROFILES.get(_uid(update), {})

def update_user_profile(update: Update, **kwargs):
    uid = _uid(update)
    cur = PROFILES.get(uid, {})
    cur.update({k: v for k, v in kwargs.items() if v is not None})
    PROFILES[uid] = cur
    _save_profiles(PROFILES)


# --- name resolution helpers (paste near other small helpers) ---
def _norm_meta_key(v: object) -> str:
    """Uppercase A–Z/0–9 only; same normalization used for COLLEGE_META_INDEX keys."""
    return re.sub(r"[^A-Z0-9]+", "", str(v or "").upper())

def resolve_college_name_from_row(row: dict) -> Optional[str]:
    """
    Try to get a displayable college name for a shortlist row.
    1) If the row already has a name column, use it.
    2) Otherwise, look up by id/code in COLLEGE_META_INDEX.
    """
    # 1) direct name on the row
    for k in ("college_name", "College Name", "name", "Institute Name"):
        val = row.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # 2) try via id/code → meta index
    for idk in ("college_id", "College ID", "college_code", "College Code", "__LookupKey"):
        key_val = row.get(idk)
        if key_val:
            meta = COLLEGE_META_INDEX.get(_norm_meta_key(key_val))
            if meta:
                for mk in ("College Name", "name", "Institute Name"):
                    mval = meta.get(mk)
                    if isinstance(mval, str) and mval.strip():
                        return mval.strip()
    return None

SHOW_INTERNAL_IDS = False 

def _format_row_plain(i: int, r: dict, *, closing_rank=None) -> str:
    """Clean one-line row used by predict. NO hostel here."""
    name  = r.get("college_name") or r.get("College Name") or "—"
    code  = r.get("college_code") or r.get("code") or r.get("college_id")
    city  = r.get("city")
    state = r.get("state") or r.get("State")

    # Fee
    fee = r.get("total_fee") or r.get("Fee")
    if isinstance(fee, (int, float)):
        fee_str = f"₹{int(fee):,}"
    elif fee in (None, "", "—"):
        fee_str = "—"
    else:
        fee_str = f"₹{fee}"

    # Closing rank (passed in or derived from record)
    cr = (closing_rank
          or r.get("ClosingRank")
          or r.get("closing")
          or r.get("closing_rank")
          or r.get("rank"))
    if cr in (None, "", "None"):
        close_str = "—"
    else:
        try:
            close_str = f"{int(float(cr)):,}"
        except Exception:
            close_str = str(cr)

    # Location label
    loc = city or state or ""
    loc_part = f", {loc}" if loc else ""

    head = f"{i}. {name}{loc_part}"
    if code:
        head += f" (`{code}`)"
    # NOTE: no “Hostel” here
    tail = f" — closing {close_str} • fee {fee_str}"
    return head + tail

# ========================= Menus & States =========================




def main_menu_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📝 Daily Quiz (Exam Mode)", callback_data="menu_quiz")],
        [InlineKeyboardButton("🏫 NEET College Predictor", callback_data="menu_predict")],
        [InlineKeyboardButton("📈 Predict from Mock Rank", callback_data="menu_mock_predict")],
        [InlineKeyboardButton("💬 Clear your NEET Doubts", callback_data="menu_ask")],
        [InlineKeyboardButton("⚙️ Setup your profile", callback_data="menu_profile")],
    ])

async def show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str = "Choose an option:"):
    explanation = (
        "📋 *Menu Options*\n\n"
        "🏫 *NEET College Predictor* – Uses your AIR & category and predict list of 10 colleges that you might get at your NEET rank based on last year's cutoffs. It will also give AI-based suggestions.\n\n"
        "🏫 *Predict from Mock Rank* – Uses your All India Mock Test Rank, Quota & Category to predict colleges based on last year's cutoffs.\n\n"
        "📝 *Daily Quiz (Exam Mode)* – Take timed quizzes and get scores.\n\n"
        "💬 *Clear your NEET Doubts* – Send text or photo to get structured solutions and follow-ups.\n\n"
        "⚙️ *Setup your profile* – Save Name, Contact, Email, Category, Domicile."
    )
    tgt = update.callback_query.message if update.callback_query else update.effective_message
    await tgt.reply_text(explanation, parse_mode="Markdown")
    await tgt.reply_text(text, reply_markup=main_menu_markup())

async def menu_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Routes top-level menu buttons to their handlers."""
    q = update.callback_query
    await q.answer()
    data = (q.data or "").strip()
    if data == "menu_quiz":
        await menu_quiz_handler(update, context)
    elif data == "menu_predict":
        # call your existing predict menu entry if you have one, else just ack
        await q.message.reply_text("Predictor coming right up. Use /predict to start.")
    elif data == "menu_mock_predict":
        await q.message.reply_text("Mock-rank predictor: use /predict to start and choose Mock Rank.")
    elif data == "menu_ask":
        await q.message.reply_text("Ask your NEET doubt with /ask.")
    elif data == "menu_profile":
        await q.message.reply_text("Open profile with /profile.")
    else:
        await q.message.reply_text("Unknown menu item.")

async def quiz_menu_router(update, context):
    q = update.callback_query
    await q.answer()
    await _safe_clear_kb(q)

    data = (q.data or "")

    if data == "quiz_mini5":
        await _start_quiz(update, context, count=5)
        return

    if data == "quiz_mini10":
        subjects = sorted({x.get("subject") for x in QUIZ_POOL if x.get("subject")})
        if not subjects:
            await q.message.reply_text("No subjects available in the quiz bank.")
            return
        rows = [[InlineKeyboardButton(s, callback_data=f"quiz_sub:{s}")] for s in subjects]
        rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu_back")])
        kb = InlineKeyboardMarkup(rows)
        await q.message.reply_text("Pick a subject for a 10-question test:", reply_markup=kb)
        return

    if data.startswith("quiz_sub:"):
        subject = data.split("quiz_sub:", 1)[1]
        await _start_quiz(update, context, count=10, subject=subject)
        return

    if data == "quiz_streaks":
        await q.message.reply_text("Streaks coming soon.")
        return

    if data == "quiz_leaderboard":
        await q.message.reply_text("Leaderboard coming soon.")
        return

async def menu_back(update, context):
    q = update.callback_query
    await q.answer()
    await _safe_clear_kb(q)
    await show_menu(update)

# States
ASK_SUBJECT = 105
ASK_WAIT = 100

ASK_AIR, ASK_QUOTA, ASK_CATEGORY, ASK_DOMICILE, ASK_PG_REQ, ASK_BOND_AVOID, ASK_PREF = range(300, 307)

PROFILE_MENU, PROFILE_SET_CATEGORY, PROFILE_SET_DOMICILE, PROFILE_SET_PREF, PROFILE_SET_EMAIL, PROFILE_SET_MOBILE, PROFILE_SET_PRIMARY = range(120, 127)



# ========================= /start & /reset =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    unlock_flow(context)

    # --- clear legacy quiz keys (old conversation-based quiz) ---
    for k in (
        "quiz_subject", "quiz_difficulty", "quiz_size", "quiz_questions", "quiz_idx",
        "quiz_correct", "quiz_wrong", "quiz_deadline", "quiz_started_at",
        "quiz_limit_secs", "quiz_wrongs_buffer", "predicted_air"
    ):
        context.user_data.pop(k, None)

    # keep Ask cleanup as you had
    for k in ("ask_subject", "ask_last_question"):
        context.user_data.pop(k, None)

    # --- clear new quiz session (button-based quiz) ---
    try:
        from bot import QUIZ_SESSIONS  # if this is the same file, remove this import line
        QUIZ_SESSIONS.pop(update.effective_user.id, None)
    except Exception:
        pass

    tgt = _target(update)
    if tgt:
        await tgt.reply_text("Welcome!", reply_markup=ReplyKeyboardRemove())
    await show_menu(update)


async def reset_lock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # wipe all per-user scratch
    for k in list(context.user_data.keys()):
        context.user_data.pop(k, None)

    # also clear new quiz session
    try:
        from bot import QUIZ_SESSIONS  # if this is the same file, remove this import line
        QUIZ_SESSIONS.pop(update.effective_user.id, None)
    except Exception:
        pass

    await update.message.reply_text("✅ State cleared. You can start fresh with /menu.")

# ========================= Profile =========================
def _canonical_category_ui(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    if t in {"general", "gen", "ur"}: return "General"
    if t == "obc": return "OBC"
    if t == "ews": return "EWS"
    if t == "sc":  return "SC"
    if t == "st":  return "ST"
    return None

def _valid_email(s: str) -> bool:
    return bool(s) and re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", s)

def _clean_mobile(s: str) -> Optional[str]:
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    return digits if 10 <= len(digits) <= 15 else None

async def setup_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # FIX: use our existing flow guard
    ok = _ensure_flow_or_bounce(update, context, "profile")
    if not ok:
        return ConversationHandler.END
    log.info("setup_profile()")

    tgt = _target(update)
    prof = get_user_profile(update)
    cat = prof.get("category", "Not set")
    dom = prof.get("domicile_state", "Not set")
    pref = prof.get("pref_type", "Not set")
    email = prof.get("email", "Not set")
    mobile = prof.get("mobile", "Not set")
    primary = prof.get("primary_id", "Not set")
    latest_air = prof.get("latest_predicted_air", "Not set")

    kb = ReplyKeyboardMarkup(
        [["Set Category", "Set Domicile"], ["Set Email", "Set Mobile"], ["Set Primary ID"], ["Show", "Done"]],
        one_time_keyboard=True, resize_keyboard=True
    )
    await tgt.reply_text(
        "👤 *Your Profile*\n"
        f"• Category: *{cat}*\n"
        f"• Domicile: *{dom}*\n"
        f"• Email: *{email}*\n"
        f"• Mobile: *{mobile}*\n"
        f"• Primary ID: *{primary}*\n"
        f"• Latest predicted AIR: *{latest_air}*\n\n"
        "_You can update/delete anytime._\n\n"
        "Choose an action:",
        parse_mode="Markdown",
        reply_markup=kb
    )
    return PROFILE_MENU

async def profile_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip().lower()
    if text == "set category":
        kb = ReplyKeyboardMarkup([["General", "OBC", "EWS", "SC", "ST"]], one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Select your category:", reply_markup=kb)
        return PROFILE_SET_CATEGORY
    elif text == "set domicile":
        kb = ReplyKeyboardMarkup([["Cancel"]], one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Type your domicile state (e.g., Delhi, Karnataka).", reply_markup=kb)
        return PROFILE_SET_DOMICILE
    elif text == "set email":
        kb = ReplyKeyboardMarkup([["Skip"]], one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Type your email (or tap Skip):", reply_markup=kb)
        return PROFILE_SET_EMAIL
    elif text == "set mobile":
        kb = ReplyKeyboardMarkup([[KeyboardButton("Share my contact", request_contact=True)], ["Skip"]],
                                 one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Send your mobile number (or tap 'Share my contact' / 'Skip'):", reply_markup=kb)
        return PROFILE_SET_MOBILE
    elif text == "set primary id":
        kb = ReplyKeyboardMarkup([["Mobile", "Email"]], one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Which should be your primary identifier?", reply_markup=kb)
        return PROFILE_SET_PRIMARY
    elif text == "show":
        return await setup_profile(update, context)
    else:
        unlock_flow(context)
        await update.message.reply_text("Profile saved. Returning to menu.", reply_markup=ReplyKeyboardRemove())
        await show_menu(update)
        return ConversationHandler.END

async def profile_set_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cat = _canonical_category_ui(update.message.text or "")
    if not cat:
        kb = ReplyKeyboardMarkup([["General", "OBC", "EWS", "SC", "ST"]], one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Pick a valid category.", reply_markup=kb)
        return PROFILE_SET_CATEGORY
    update_user_profile(update, category=cat)
    await update.message.reply_text(f"Saved category: *{cat}*", parse_mode="Markdown")
    return await setup_profile(update, context)

async def profile_set_domicile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip()
    if txt.lower() == "cancel" or not txt:
        await update.message.reply_text("Cancelled.", reply_markup=ReplyKeyboardRemove())
        return await setup_profile(update, context)
    update_user_profile(update, domicile_state=txt)
    await update.message.reply_text(f"Saved domicile state: *{txt}*", parse_mode="Markdown")
    return await setup_profile(update, context)

def weights_for_profile(label: str) -> Dict[str, float]:
    label = (label or "").strip().lower()
    if label == "low fee":
        return {"nirf": 0.25, "fee": 0.55, "safety": 0.12, "pg": 0.04, "bond": 0.02, "location": 0.02}
    if label == "top ranked":
        return {"nirf": 0.60, "fee": 0.20, "safety": 0.10, "pg": 0.03, "bond": 0.02, "location": 0.05}
    if label == "safety first":
        return {"nirf": 0.20, "fee": 0.20, "safety": 0.50, "pg": 0.05, "bond": 0.02, "location": 0.03}
    return {"nirf": 0.40, "fee": 0.30, "safety": 0.12, "pg": 0.06, "bond": 0.04, "location": 0.08}

def weights_for_label(label: str):
    # Backward compatible alias if older code still calls this name
    return weights_for_profile(label)


from telegram.constants import ChatAction

def _yn(x):
    s = str(x).strip().lower() if x is not None else ""
    return "yes" if s in ("1","true","yes","y") else "unknown"

def _pick(d: dict, *keys):
    for k in keys:
        if k in d:
            v = d.get(k)
            if v not in (None, "", "—"):
                return v
    return None

#New Quiz----------

def _quiz_subjects_from_pool() -> list[str]:
    """Find unique subjects available in QUIZ_POOL. Fallback to standard list."""
    try:
        subs = sorted({str(q.get("subject") or "").strip() for q in QUIZ_POOL if q.get("subject")})
        subs = [s for s in subs if s]  # remove blanks
        return subs or ["Physics", "Chemistry", "Zoology", "Botany"]
    except Exception:
        return ["Physics", "Chemistry", "Zoology", "Botany"]



async def menu_quiz_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    # Offer a quick choice between 5 or 10
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("5 Questions",  callback_data="start_quiz:5")],
        [InlineKeyboardButton("10 Questions", callback_data="start_quiz:10")],
    ])
    await q.edit_message_text("Choose a quiz size:", reply_markup=kb)

async def menu_quiz_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    _, count_s = q.data.split(":")
    count = int(count_s)
    # start the new quiz with defaults (no subject/difficulty filter)
    await _start_quiz(update, context, count=count)

async def menu_quiz_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    await _safe_clear_kb(q)
    
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🎯 Mini Quiz (5)", callback_data="quiz:mini5")],
        [InlineKeyboardButton("📚 Mini Test (10, choose subject)", callback_data="quiz:mini10")],
        [InlineKeyboardButton("🔥 Streaks", callback_data="quiz:streaks")],
        [InlineKeyboardButton("🏆 Leaderboard", callback_data="quiz:leaderboard")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:back")],
    ])

    await _safe_set_kb(q, kb)
    await q.message.reply_text("Choose a quiz mode:", reply_markup=kb)


async def ai_notes_from_shortlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import math, os


async def quiz_start_mini5(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _start_quiz(update, context, count=5)

async def quiz_pick_subject(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if q:
        await q.answer()
        tgt = q.message
    else:
        tgt = update.effective_message

    subjects = ["Physics", "Chemistry", "Botany", "Zoology"]
    kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton(s, callback_data=f"quiz:subject:{s}")] for s in subjects]
    )
    await tgt.reply_text("Pick a subject for a 10-question test:", reply_markup=kb)

async def quiz_start_subject10(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if q:
        await q.answer()
        data = q.data or ""
        subject = data.split(":", 2)[2] if data.count(":") >= 2 else None
        if not subject:
            await q.message.reply_text("Subject missing. Please tap a subject again.")
            return
    else:
        await update.effective_message.reply_text("Use the buttons to choose a subject.")
        return

    await _start_quiz(update, context, count=10, subject=subject)

# Optional stubs
async def quiz_streaks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.callback_query:
        await update.callback_query.answer()
    await (update.effective_message or update.callback_query.message).reply_text(
        "Streaks coming soon 🔜"
    )

async def quiz_leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.callback_query:
        await update.callback_query.answer()
    await (update.effective_message or update.callback_query.message).reply_text(
        "Leaderboard coming soon 🔜"
    )



async def menu_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = (q.data or "")
    if data == "menu_quiz":
        # open the quiz picker (Mini 5 / Mini 10 + subject)
        await show_quiz_menu(update, context)
    elif data == "menu_predict":
        await predict_start(update, context)          # your existing entry
    elif data == "menu_mock_predict":
        await mock_predict_start(update, context)     # your existing entry
    elif data == "menu_ask":
        await ask_start(update, context)              # your existing entry
    elif data == "menu_profile":
        await profile_start(update, context)          # your existing entry

    
#----------------------------New Quiz end
    # ---------------- small helpers ----------------
    NA_STRINGS = {"", "—", "-", "na", "n/a", "nan", "none", "null"}

    def _safe_str(v, default: str = "") -> str:
        try:
            if v is None: return default
            if isinstance(v, float) and math.isnan(v): return default
            s = str(v).strip()
            return default if s.lower() in NA_STRINGS else s
        except Exception:
            return default

    def _pick(d, *keys):
        if not isinstance(d, dict):
            return None
        for k in keys:
            v = d.get(k)
            if v not in (None, "", "—"):
                return v
        return None

    def _is_missing(v):
        try:
            if v is None: return True
            if isinstance(v, float) and math.isnan(v): return True
            return str(v).strip().lower() in NA_STRINGS
        except Exception:
            return True

    def _fmt_money(v):
        try:
            s = str(v).replace(",", "").strip()
            if not s or s.lower() in NA_STRINGS: return "—"
            n = float(s)
            return f"₹{int(n):,}"
        except Exception:
            return "—"

    def _fmt_rank_val(v):
        try:
            s = str(v).strip()
            if _is_missing(s): return "—"
            return f"{int(float(s)):,}"
        except Exception:
            return "—"

    def _yn(v):
        if _is_missing(v): return "—"
        s = str(v).strip().lower()
        if s in {"yes", "y", "true", "1", "available"}: return "Yes"
        if s in {"no", "n", "false", "0", "not available"}: return "No"
        return str(v)

    def _fmt_bond_line(bond_years, bond_penalty_lakhs):
        def _num(x):
            try:
                if _is_missing(x): return None
                return float(str(x).replace(",", "").strip())
            except Exception:
                return None
        yrs = _num(bond_years)
        pen = _num(bond_penalty_lakhs)
        if not yrs and not pen:
            return "No"
        parts = []
        if yrs and yrs > 0:
            parts.append(f"{int(yrs) if float(yrs).is_integer() else yrs} yrs")
        if pen and pen > 0:
            lakhs = int(pen) if float(pen).is_integer() else pen
            parts.append(f"₹{lakhs}L")
        return ", ".join(parts) if parts else "No"

    def _city_vibe_from_row(city: str, state: str) -> str:
        c = (city or "").strip().lower()
        s = (state or "").strip().lower()
        metro = {"delhi", "new delhi", "mumbai", "kolkata", "chennai", "bengaluru", "bangalore", "hyderabad", "pune"}
        if c in metro:
            return "Metro pace; higher living costs; English/Hindi widely used"
        if s in {"kerala", "goa"}:
            return "Calmer pace; mid living costs; local language common"
        return "Calmer pace; mid living costs; local language common"

    def _why_from_signals(name, ownership, pg_quota, bond_years, hostel_avail) -> str:
        own = (ownership or "").lower()
        nm  = (name or "").upper()
        if "AIIMS" in nm:
            base = "Central government AIIMS"
        elif "JIPMER" in nm or "PGIMER" in nm:
            base = "Central government INI"
        elif "government" in own or "state" in own or "gov" in own:
            base = "State government medical college"
        elif "deemed" in own or "private" in own:
            base = "Private/deemed medical college"
        else:
            base = "Reputed medical college"

        extras = []
        if _yn(pg_quota) == "Yes":
            extras.append("PG quota available")
        try:
            yrs = float(str(bond_years).strip()) if not _is_missing(bond_years) else 0.0
        except Exception:
            yrs = 0.0
        extras.append(f"bond {int(yrs) if yrs and float(yrs).is_integer() else yrs} yrs" if yrs > 0 else "no/low bond")
        hv = _yn(hostel_avail)
        if hv in {"Yes", "No"}:
            extras.append(f"hostel {hv}")
        return base + ("; " + "; ".join(extras) if extras else "")

    def _resolve_master_row(r, idx_by_code, idx_by_id, idx_by_name):
        for key in ("college_code", "code", "college_id", "institute_code"):
            v = r.get(key) if isinstance(r, dict) else None
            if not _is_missing(v):
                sid = str(v).strip()
                if sid in idx_by_code: return idx_by_code[sid]
                if sid in idx_by_id:   return idx_by_id[sid]
        nm = _safe_str(_pick(r, "college_name", "College Name"))
        return idx_by_name.get(nm.lower(), {})

    # ---------------- handler body ----------------
    try:
        if update.callback_query:
            await update.callback_query.answer()
        chat_id = update.effective_chat.id
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        status = await context.bot.send_message(chat_id=chat_id, text="🧠 Working on AI notes for your top colleges…")

        user = context.user_data or {}
        items = context.user_data.get("LAST_SHORTLIST") or context.user_data.get("last_shortlist") or []
        if not items:
            user["require_pg_quota"] = None
            user["avoid_bond"] = None
            items = shortlist_and_score(COLLEGES, user, cutoff_lookup=CUTOFF_LOOKUP) or []
            items = _dedupe_results(items)
        items = items[:10]
        if not items:
            await status.edit_text("I couldn’t find any shortlisted colleges to summarize. Run /predict first.")
            return

        # Build master indexes
        try:
            master_rows = COLLEGES.to_dict("records") if hasattr(COLLEGES, "to_dict") else (COLLEGES if isinstance(COLLEGES, list) else [])
        except Exception:
            master_rows = COLLEGES if isinstance(COLLEGES, list) else []
        idx_by_code, idx_by_id, idx_by_name = {}, {}, {}
        for row in master_rows:
            if not isinstance(row, dict): continue
            code = _safe_str(row.get("college_code") or row.get("code"))
            cid  = _safe_str(row.get("college_id")  or row.get("institute_code"))
            nm   = _safe_str(row.get("college_name") or row.get("College Name")).lower()
            if code: idx_by_code[code] = row
            if cid:  idx_by_id[cid] = row
            if nm:   idx_by_name[nm] = row

        # context for closing-rank lookup
        round_ui = user.get("cutoff_round") or user.get("round") or "2025_R1"
        quota    = user.get("quota") or "AIQ"
        category = user.get("category") or "General"
        df_lookup = context.application.bot_data.get("CUTOFFS_DF")

        # Normalize & prepare blocks
        blocks = []
        for i, r in enumerate(items, 1):
            if not isinstance(r, dict): continue
            m = _resolve_master_row(r, idx_by_code, idx_by_id, idx_by_name)  # safe {}

            name  = _safe_str(_pick(r, "college_name", "College Name") or _pick(m, "college_name", "College Name")) or "Unknown college"
            state = _safe_str(_pick(r, "state", "State")               or _pick(m, "state", "State"))
            city  = _safe_str(_pick(r, "city", "City")                 or _pick(m, "city", "City"))
            place = ", ".join([x for x in (city, state) if x])

            closing = r.get("ClosingRank") or r.get("closing") or r.get("rank")
            if _is_missing(closing):
                ids = [r.get("college_code"), r.get("code"), r.get("college_id"), r.get("institute_code"), name]
                closing = _closing_rank_for_identifiers(
                    [x for x in ids if not _is_missing(x)],
                    round_ui, quota, category,
                    df_lookup=df_lookup, lookup_dict=CUTOFF_LOOKUP
                )

            fee          = _pick(r, "total_fee", "Fee") or _pick(m, "total_fee", "Fee")
            ownership    = r.get("ownership") or m.get("ownership")
            pg_quota     = r.get("pg_quota")  or m.get("pg_quota")
            bond_years   = r.get("bond_years") if not _is_missing(r.get("bond_years")) else m.get("bond_years")
            bond_penalty = r.get("bond_penalty_lakhs") if not _is_missing(r.get("bond_penalty_lakhs")) else m.get("bond_penalty_lakhs")
            hostel_avail = r.get("hostel_available") if not _is_missing(r.get("hostel_available")) else m.get("hostel_available")

            header   = f"{i}. {name}" + (f", {place}" if place else "")
            rank_ln  = f"Closing Rank: {_fmt_rank_val(closing)}"
            fee_ln   = f"Annual Fee: {_fmt_money(fee)}"
            why_ln   = "Why it stands out: " + _why_from_signals(name, ownership, pg_quota, bond_years, hostel_avail)
            vibe_ln  = "City & campus vibe: " + _city_vibe_from_row(city, state)
            pg_ln    = f"PG Quota: {_yn(pg_quota)}"
            bond_ln  = f"Bond: {_fmt_bond_line(bond_years, bond_penalty)}"
            hostel_ln= f"Hostel: {_yn(hostel_avail)}"

            blocks.append("\n".join([header, rank_ln, fee_ln, why_ln, vibe_ln, pg_ln, bond_ln, hostel_ln]))

        # Always send the rich offline text (predictable format)
        ai_text = "\n\n".join(blocks)
        await status.edit_text(ai_text)

    except Exception:
        log.exception("[ai_notes] failed")
        try:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="❌ Couldn't prepare AI notes.")
        except Exception:
            pass







async def profile_set_pref(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pref = (update.message.text or "").strip()
    if pref not in {"Balanced", "Low Fee", "Top Ranked", "Safety First"}:
        kb = ReplyKeyboardMarkup([["Balanced", "Low Fee", "Top Ranked", "Safety First"]], one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Pick a valid preference.", reply_markup=kb)
        return PROFILE_SET_PREF
    update_user_profile(update, pref_type=pref)
    await update.message.reply_text(f"Saved preference: *{pref}*", parse_mode="Markdown")
    return await setup_profile(update, context)

async def profile_set_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip()
    if txt.lower() == "skip":
        update_user_profile(update, email=None)
        await update.message.reply_text("Skipped email.")
        return await setup_profile(update, context)
    if not _valid_email(txt):
        kb = ReplyKeyboardMarkup([["Skip"]], one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Invalid email. Try again or tap Skip.", reply_markup=kb)
        return PROFILE_SET_EMAIL
    update_user_profile(update, email=txt)
    await update.message.reply_text(f"Saved email: *{txt}*", parse_mode="Markdown")
    return await setup_profile(update, context)

async def profile_set_mobile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    contact = update.message.contact
    if contact and contact.phone_number:
        mobile = _clean_mobile(contact.phone_number)
        if mobile:
            update_user_profile(update, mobile=mobile)
            await update.message.reply_text(f"Saved mobile: *{mobile}*", parse_mode="Markdown")
            return await setup_profile(update, context)
    txt = (update.message.text or "").strip()
    if txt.lower() == "skip":
        update_user_profile(update, mobile=None)
        await update.message.reply_text("Skipped mobile.")
        return await setup_profile(update, context)
    mobile = _clean_mobile(txt)
    if not mobile:
        kb = ReplyKeyboardMarkup([[KeyboardButton("Share my contact", request_contact=True)], ["Skip"]],
                                 one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Invalid number. Share contact or type again (10–15 digits).", reply_markup=kb)
        return PROFILE_SET_MOBILE
    update_user_profile(update, mobile=mobile)
    await update.message.reply_text(f"Saved mobile: *{mobile}*", parse_mode="Markdown")
    return await setup_profile(update, context)

async def profile_set_primary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    choice = (update.message.text or "").strip().lower()
    if choice not in {"mobile", "email"}:
        kb = ReplyKeyboardMarkup([["Mobile", "Email"]], one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Pick Mobile or Email.", reply_markup=kb)
        return PROFILE_SET_PRIMARY
    update_user_profile(update, primary_id=choice)
    await update.message.reply_text(f"Saved primary identifier: *{choice.title()}*", parse_mode="Markdown")
    return await setup_profile(update, context)

# ========================= OpenAI (Ask / Doubt) =========================

NEET_TUTOR_SYSTEM = (
    "You are a NEET UG tutor. Solve only NEET-syllabus style questions. "
    "Keep solutions concise and structured:\n"
    "• Given (symbols & data)\n"
    "• Approach (1–2 lines)\n"
    "• Steps (numbered)\n"
    "• Final answer (with units if applicable)\n"
    "If MCQ, also add: Why other options are wrong (1 line each). "
    "Physics: write formula first & check units. "
    "Chemistry: mention reagent/mechanism. "
    "Biology: prefer NCERT terminology and a memory hook. "
    "If the image is unclear, ask for a clearer photo."
)

COUNSELLING_SYSTEM = (
    "You are a NEET counselling assistant. Give clear, step-by-step answers about counselling rules, "
    "eligibility, quotas, round flow, upgradation, resignation, bond/fee policies, state vs AIQ, and documentation. "
    "Keep answers concise, bullet-first, and cite round names consistently (R1/R2/Mop-up/Stray). "
    "Avoid legal/financial advice; defer to official MCC or State Board notices when uncertain."
)

def _subject_hint_text(subj: Optional[str]) -> str:
    if not subj:
        return ""
    s = subj.lower()
    if s == "physics":
        return "Subject = Physics. Emphasize formulas, knowns/unknowns, units."
    if s == "chemistry":
        return "Subject = Chemistry. Specify topic (Physical/Organic/Inorganic) if clear."
    if s == "zoology":
        return "Subject = Zoology. Use NCERT terms and concise definitions."
    if s == "botany":
        return "Subject = Botany. Use NCERT terms and concise definitions."
    return ""

async def _ai_followup(mode: str, *, subject: str | None, concept: str) -> str:
    """
    mode: 'similar' | 'explain' | 'flash'
    Returns plain text (no Markdown/LaTeX).
    """
    subj = (subject or "NEET").strip()
    base_rules = (
        "Answer in concise plain text (no Markdown/LaTeX/code fences). "
        "Use short lines and simple bullets like '•' if needed."
    )

    if mode == "similar":
        prompt = (
            f"{base_rules}\n"
            f"Create ONE {subj} question similar to: \"{concept}\".\n"
            "If possible make it an MCQ with four options A–D.\n"
            "Return:\n"
            "Q: <question>\n"
            "Options: A) ..  B) ..  C) ..  D) ..\n"
            "Answer: <option letter>\n"
            "Solution: 4–6 lines covering approach + a common trap."
        )
    elif mode == "explain":
        prompt = (
            f"{base_rules}\n"
            f"Explain the core concept(s) behind: \"{concept}\" for {subj}.\n"
            "Keep it crisp: 5–8 lines, include 2–3 memory hooks and 1 common mistake."
        )
    else:  # flash
        prompt = (
            f"{base_rules}\n"
            f"Make 5 very concise {subj} flashcards based on: \"{concept}\".\n"
            "Format each as: Q: <prompt> | A: <short answer>"
        )

    return await call_openai(prompt)

async def call_openai(prompt: str, *, model: str | None = None, max_output_tokens: int = 600) -> str:
    """
    Minimal REST client for the Responses API that works with raw JSON.
    Gathers assistant text from data["output"][*]["content"][*]["text"].
    """
    key = OPENAI_API_KEY
    if not key or not key.startswith("sk-"):
        log.error("ask_followup: OPENAI_API_KEY missing/invalid")
        return "Sorry—AI isn’t configured yet."

    payload = {
        "model": model or OPENAI_MODEL,
        # Responses API expects 'input' as either a string or structured messages.
        # A plain string is fine for simple text prompts:
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "temperature": 0.3,
        # Small style hint; optional:
        "instructions": "You are a concise NEET helper. Use plain text only (no Markdown/LaTeX).",
    }
    headers = {"Authorization": f"Bearer {key}"}

    try:
        async with httpx.AsyncClient(timeout=60) as cli:
            r = await cli.post(f"{OPENAI_BASE_URL.rstrip('/')}/responses",
                               headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        # 1) SDK convenience field (won't exist in raw REST, but keep just in case)
        txt = (data.get("output_text") or "").strip()

        # 2) Robust fallback: collect from the generic 'output' array
        if not txt:
            parts = []
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        # typical shape: {"type": "output_text", "text": "..."}
                        t = c.get("text")
                        if t:
                            parts.append(t)
            txt = "\n".join(parts).strip()

        return txt or "I couldn’t generate a response."
    except httpx.HTTPStatusError as e:
        # Surface useful error details
        err_body = ""
        try:
            err_body = e.response.text[:400]
        except Exception:
            pass
        log.exception("OpenAI HTTP error: %s %s", e, err_body)
        return f"Sorry—couldn’t generate that. (HTTP {e.response.status_code})"
    except Exception as e:
        log.exception("ask_followup: OpenAI error")
        return f"Sorry—couldn’t generate that. ({type(e).__name__})"
        
async def quota_counts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rk = context.user_data.get("cutoff_round", ACTIVE_CUTOFF_ROUND_DEFAULT)
    per = CUTOFFS_Q.get(rk, {})
    lines = [f"Round {rk} — quota buckets loaded:"]
    for qname, qmap in sorted(per.items()):
        lines.append(f"• {qname}: {len(qmap)} colleges")
    await update.message.reply_text("\n".join(lines))
    

async def ask_openai_vision(
    question: str,
    image_path: Optional[str] = None,
    subject_hint: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Calls OpenAI once; returns (ok, html_text).
    Produces clean HTML tailored for Telegram (no markdown/latex).
    """
    import asyncio, base64, html

    cli = _ensure_openai_client()  # make sure you added the helper shown earlier

    system = (
         COUNSELLING_SYSTEM if (subject_hint and subject_hint.strip().lower()=="counselling") else
         "You are a NEET helper. Answer concisely in plain text.\n"
         "Avoid Markdown and LaTeX. Use short paragraphs and simple bullets like '• '.\n"
         "Bold section titles using <b>…</b>. Use <br> for line breaks only if necessary.\n"
         "Never include code fences or math delimiters." 
    )
    if subject_hint:
        system += f"\nSubject focus: {subject_hint}."

    # --- Build OpenAI Responses input with correct content types ---
    parts: list[dict] = []
    if image_path:
        # Encode image as base64 and send as input_image
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        parts.append({"type": "input_image", "image_data": b64})

    # Always include user's text as input_text
    parts.append({"type": "input_text", "text": question})

    # Run in a thread so the bot loop doesn’t block
    loop = asyncio.get_running_loop()

    
    def _format_quick_qna(items: list[dict]) -> str:
        blocks = []
        for i, qa in enumerate(items, 1):
            q = html.escape((qa.get("q") or "").strip())
            a = html.escape((qa.get("a") or "").strip())
            if q and a:
                blocks.append(f"<b>Q{i}.</b> {q}\n<b>Ans:</b> {a}")
        return "\n\n".join(blocks) if blocks else "No questions generated."
    
    
    def _call_sync() -> str:
        resp = cli.responses.create(
            model="gpt-4o-mini",  # keep your current model
            input=[{"role": "user", "content": parts}],
            instructions=system,
            temperature=0.3,
        )
        # Prefer convenience accessor; fall back to assembling from output events
        if getattr(resp, "output_text", None):
            return resp.output_text.strip()
        out_chunks = []
        for item in getattr(resp, "output", []):
            if getattr(item, "type", "") == "output_text":
                out_chunks.append(item.text)
        return "\n".join(out_chunks).strip()

    try:
        raw = await loop.run_in_executor(None, _call_sync)
        html_text = _to_safe_html(raw)  # your existing sanitizer
        return True, html_text
    except Exception as e:
        return False, (
            "Sorry—couldn’t process that right now.<br>"
            f"<code>{html.escape(str(e))}</code>"
        )




def ask_subject_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [["Physics", "Chemistry"], ["Zoology", "Botany"], ["Counselling"], ["Skip", "Cancel"]],
        one_time_keyboard=True, resize_keyboard=True
    )

def _closing_rank_smart(code, quota, category, round_code, df_lookup, lookup_dict):
    """
    Robust closing-rank resolver.
    1) Try CUTOFFS_DF (if available).
    2) Scan CUTOFF_LOOKUP dict in many shapes:
       - keys as tuples: (round, quota, category, code) or (round, quota, code)
       - keys as pipe strings: "2025_R1|AIQ|OP|C0065"
       - nested dicts: lookup_dict[code][round][quota][category] -> {ClosingRank: ...}
       - list of triplets/dicts per code.
    Returns int (preferred), str, or None.
    """
    if not code:
        return None

    # --- 0) base normalizers & candidate aliases
    def _norm_q(q):
        if not q: return None
        s = str(q).strip()
        return {"Central": "AIQ", "All India Quota": "AIQ"}.get(s, s)

    def _norm_c(c):
        if not c: return None
        s = str(c).strip()
        return {"General": "OP", "Open": "OP", "UR": "OP"}.get(s, s)

    code_s = str(code).strip()
    q_norm  = _norm_q(quota)
    c_norm  = _norm_c(category)

    q_aliases = list(dict.fromkeys([q_norm, quota, "AIQ", "Central", None]))
    c_aliases = list(dict.fromkeys([c_norm, category, "OP", "General", None]))
    r_aliases = list(dict.fromkeys([round_code, None]))  # try specific round then any

    # --- 1) DataFrame path (fast)
    try:
        df = df_lookup
        if df is not None and hasattr(df, "empty") and not df.empty:
            sub = df.copy()
            sub["college_code"] = sub["college_code"].astype(str)
            sub = sub[sub["college_code"] == code_s]
            # Try progressively stricter → looser filters
            for r_try in r_aliases:
                sub_r = sub if r_try is None else sub[sub["round_code"] == r_try] if "round_code" in sub.columns else sub
                for q_try in q_aliases:
                    sub_q = sub_r if q_try is None else sub_r[sub_r["quota"] == q_try] if "quota" in sub_r.columns else sub_r
                    for c_try in c_aliases:
                        sub_c = sub_q if c_try is None else sub_q[sub_q["category"] == c_try] if "category" in sub_q.columns else sub_q
                        if "ClosingRank" in sub_c.columns and not sub_c["ClosingRank"].dropna().empty:
                            val = sub_c["ClosingRank"].dropna().iloc[0]
                            try:   return int(float(val))
                            except: return val
    except Exception:
        pass

    # --- 2) Dictionary path (very tolerant)
    d = lookup_dict or {}

    def _extract_cr(v):
        # single value
        if isinstance(v, (int, float, str)):
            return v
        # dict with common fields
        if isinstance(v, dict):
            for key in ("ClosingRank", "closing", "closing_rank", "rank", "cr"):
                if key in v and v[key] not in (None, "", "None"):
                    return v[key]
        return None

    def _score_and_pick(vals):
        """
        Choose the 'best' closing rank:
        - prefer numeric; among numeric, smallest wins
        - else return first non-empty string
        """
        best_num = None
        best_str = None
        for v in vals:
            if v is None: 
                continue
            try:
                n = int(float(v))
                if best_num is None or n < best_num:
                    best_num = n
            except Exception:
                if best_str is None:
                    best_str = str(v)
        return best_num if best_num is not None else best_str

    found = []

    # 2a) Flat keys (tuple or pipe-joined)
    try:
        for k, v in d.items():
            rd = qd = cd = kd = None
            if isinstance(k, tuple):
                parts = list(k)
                # try to map common tuple layouts
                if len(parts) >= 4:
                    rd, qd, cd, kd = parts[:4]
                elif len(parts) == 3:
                    rd, qd, kd = parts
                elif len(parts) == 2:
                    kd, rd = parts
            elif isinstance(k, str) and "|" in k:
                parts = k.split("|")
                # try variants with 4, 3, or 2 fields
                if len(parts) >= 4:
                    rd, qd, cd, kd = parts[:4]
                elif len(parts) == 3:
                    rd, qd, kd = parts
                elif len(parts) == 2:
                    kd, rd = parts
            else:
                # sometimes top-level key is just the college code
                if str(k).strip() == code_s:
                    kd = k

            if kd is None or str(kd).strip() != code_s:
                continue

            for r_try in r_aliases:
                # hard check on round if both present
                if r_try is not None and rd is not None and str(rd) != str(r_try):
                    continue
                for q_try in q_aliases:
                    if q_try is not None and qd is not None and str(qd) != str(q_try):
                        continue
                    for c_try in c_aliases:
                        if c_try is not None and cd is not None and str(cd) != str(c_try):
                            continue
                        cr = _extract_cr(v)
                        if cr is not None:
                            found.append(cr)
    except Exception:
        pass

    # 2b) Nested dicts under code
    try:
        sub = d.get(code_s)
        if isinstance(sub, dict):
            # shapes like sub[round][quota][category] -> {...}
            for r_try in r_aliases:
                layers_r = [r_try] if r_try in sub else list(sub.keys()) if r_try is None else [r_try]
                for rkey in layers_r:
                    node_r = sub.get(rkey) if isinstance(sub, dict) else None
                    if not isinstance(node_r, dict): 
                        cr = _extract_cr(node_r)
                        if cr is not None: found.append(cr); continue
                    for q_try in q_aliases:
                        node_q = None
                        if isinstance(node_r, dict):
                            node_q = node_r.get(q_try) if q_try in node_r else node_r
                        if node_q is None: 
                            continue
                        if not isinstance(node_q, dict):
                            cr = _extract_cr(node_q)
                            if cr is not None: found.append(cr); continue
                        for c_try in c_aliases:
                            node_c = node_q.get(c_try) if isinstance(node_q, dict) and c_try in node_q else node_q
                            cr = _extract_cr(node_c)
                            if cr is not None:
                                found.append(cr)
        # lists under code
        if isinstance(sub, list):
            for item in sub:
                cr = _extract_cr(item)
                if cr is not None:
                    found.append(cr)
    except Exception:
        pass

    return _score_and_pick(found) if found else None

def _closing_rank_for(code, quota, category, round_code, df_lookup):
    """Try hard to get a closing rank for a college given user context."""
    import pandas as pd  # local import to avoid top-level cycles
    if code is None:
        return None
    if df_lookup is None or getattr(df_lookup, "empty", True):
        # best-effort: rebuild quickly from CUTOFF_LOOKUP if we can
        try:
            raw = _flatten_cutoff_lookup_to_df(globals().get("CUTOFF_LOOKUP") or {})
            df_lookup = _normalize_cutoffs_df(raw)
        except Exception:
            return None
    try:
        sub = df_lookup.copy()
        sub["college_code"] = sub["college_code"].astype(str)
        sub = sub[sub["college_code"] == str(code)]
        if "quota" in sub.columns and quota:
            sub = sub[sub["quota"] == quota]
        if "category" in sub.columns and category:
            sub = sub[sub["category"] == category]
        if "round_code" in sub.columns and round_code:
            sub = sub[sub["round_code"] == round_code]
        if "ClosingRank" in sub.columns:
            vals = sub["ClosingRank"].dropna()
            if not vals.empty:
                try:
                    return int(float(vals.iloc[0]))
                except Exception:
                    return None
        return None
    except Exception:
        return None

async def ask_more_flashcards(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    # Assume `cards` is already prepared or generated before this point
    formatted = format_flashcards(cards)
    await update.callback_query.message.reply_text(formatted, parse_mode="HTML")


async def ask_more_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from telegram.constants import ChatAction
    import contextlib

    q = update.callback_query
    data = (q.data or "").strip()
    await q.answer()

    subject = context.user_data.get("ask_subject")
    concept = context.user_data.get("ask_last_question")

    if data == "ask_more:quickqa":
        with contextlib.suppress(Exception):
            await q.message.edit_reply_markup(reply_markup=None)
        with contextlib.suppress(Exception):
            await q.message.chat.send_action(action=ChatAction.TYPING)

        ok, text = await _gen_quick_qna(subject=subject, concept=concept, n=5)
        txt = text if ok else "Couldn’t generate practice questions this time."

        for i in range(0, len(txt), 3800):
            await update.effective_chat.send_message(
                txt[i:i+3800], parse_mode="HTML", disable_web_page_preview=True
            )
        return

async def guard_or_block(update: Update, context: ContextTypes.DEFAULT_TYPE, want: str) -> bool:
    """
    Convenience: ensure only one flow runs at a time.
    Returns True if we can proceed; False if the user must /cancel the other flow.
    """
    blocked = _start_flow(context, want)
    if blocked and blocked != want:
        tgt = update.callback_query.message if getattr(update, "callback_query", None) else update.message
        if tgt:
            try:
                await tgt.reply_text(
                    f"You're currently in *{blocked}*. Send /cancel to exit before starting *{want}*.",
                    parse_mode="Markdown"
                )
            except Exception:
                pass
        return False
    return True


def tri_inline(prefix: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Yes", callback_data=f"{prefix}:yes"),
         InlineKeyboardButton("No", callback_data=f"{prefix}:no")],
        [InlineKeyboardButton("No preference", callback_data=f"{prefix}:any")],
    ])


async def ask_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await guard_or_block(update, context, "ask")
    if not ok:
        return ConversationHandler.END
    log.info("ask_start()")
    context.user_data.pop("ask_subject", None)
    context.user_data.pop("ask_last_question", None)
    tgt = _target(update)
    await tgt.reply_text(
        "💬 *Ask a NEET doubt*\nFirst, pick a subject (optional).",
        parse_mode="Markdown",
        reply_markup=ask_subject_keyboard()
    )
    return ASK_SUBJECT


async def ask_subject_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip().lower()
    if text == "cancel":
        unlock_flow(context)
        await update.message.reply_text("Ask cancelled.", reply_markup=ReplyKeyboardRemove())
        await show_menu(update)
        return ConversationHandler.END
    if text != "skip" and text not in {"physics", "chemistry", "zoology", "botany", "counselling"}:
        await update.message.reply_text("Choose from buttons (or Skip).", reply_markup=ask_subject_keyboard())
        return ASK_SUBJECT

    context.user_data["ask_subject"] = None if text == "skip" else text.title()
    await update.message.reply_text(
        "Send your question as *text* OR upload a *photo* (book scan / handwritten). "
        "Optionally add a short caption like “kinematics MCQ”."
        "For *Counselling* queries, include quota, category, rank, and state if relevant.",

        parse_mode="Markdown",
        reply_markup=ReplyKeyboardRemove()
    )
    return ASK_WAIT

async def _gen_quick_qna(*, subject: str | None, concept: str | None, n: int = 5) -> tuple[bool, str]:
    """
    Returns (ok, html_text) with exactly n Q&A items.
    Uses plain text -> converted to simple HTML (no <br>).
    """
    import asyncio, html

    cli = _ensure_openai_client()

    focus = (subject or "").strip()
    topic = (concept or "").strip()
    n = max(3, min(10, int(n or 5)))  # clamp 3..10

    sys_prompt = (
        "You generate compact NEET-style practice items.\n"
        "Output exactly N question–answer pairs as plain text.\n"
        "Constraints:\n"
        "• No Markdown, no LaTeX, no code fences.\n"
        "• Keep each Q and A to 1–2 lines.\n"
        "• Number them 1..N.\n"
        "Format strictly:\n"
        "1) Q: <question>\n"
        "   A: <answer>\n"
        "2) Q: <question>\n"
        "   A: <answer>\n"
        "...\n"
    )
    user_prompt = f"N={n}\nSubject={focus or 'NEET'}\nConcept={topic or 'core fundamentals'}"

    loop = asyncio.get_running_loop()

    def _call():
        resp = cli.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
            instructions=sys_prompt,
        )
        return resp.output_text or ""

    try:
        raw = await loop.run_in_executor(None, _call)
    except Exception as e:
        return False, f"Couldn’t generate practice items right now. <code>{html.escape(str(e))}</code>"

    # Parse lines -> enforce 1..N blocks
    lines = [l.rstrip() for l in raw.splitlines() if l.strip()]
    blocks: list[tuple[str, str]] = []
    cur_q = None
    cur_a = None
    for line in lines:
        # normalize like "1) Q: ..." or "1. Q: ..." or "Q:"
        l = line.strip()
        if l[:2].isdigit() or l.startswith(("Q:", "Q ")):
            if "Q:" in l:
                # start new block
                if cur_q is not None and cur_a is not None:
                    blocks.append((cur_q, cur_a))
                cur_q = l.split("Q:", 1)[1].strip()
                cur_a = None
                continue
        if l.startswith("A:"):
            cur_a = l.split("A:", 1)[1].strip()
            continue
        # If line continues Q or A
        if cur_a is None and cur_q is not None:
            cur_q = (cur_q + " " + l).strip()
        elif cur_a is not None:
            cur_a = (cur_a + " " + l).strip()

    if cur_q is not None and cur_a is not None:
        blocks.append((cur_q, cur_a))

    # If parsing is weak, just fall back to raw
    if not blocks:
        safe = html.escape(raw)
        safe = safe.replace("\n", "\n")  # keep as lines; no <br> in Telegram
        return True, f"<b>Quick Q&A</b>\n{safe}"

    # Trim to n
    blocks = blocks[:n]

    # Build HTML (no <br>, just \n)
    out_lines = ["<b>Quick Q&A</b>"]
    for i, (q, a) in enumerate(blocks, 1):
        out_lines.append(f"{i}) <b>Q:</b> {html.escape(q)}")
        out_lines.append(f"    <b>A:</b> {html.escape(a)}")
    return True, "\n".join(out_lines)

def _ask_followup_markup():
    from telegram import InlineKeyboardMarkup, InlineKeyboardButton
    rows = [
        [InlineKeyboardButton("🔁 Similar question", callback_data="ask_more:similar")],
        [InlineKeyboardButton("📚 Explain concept",  callback_data="ask_more:explain")],
        [InlineKeyboardButton("🧠 Quick Q&A (5)",   callback_data="ask_more:quickqa")],  # <—
    ]
    return InlineKeyboardMarkup(rows)
        


async def ask_more_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from telegram.constants import ChatAction
    import contextlib

    q = update.callback_query
    data = (q.data or "").strip()
    await q.answer()

    subject = context.user_data.get("ask_subject")
    concept = context.user_data.get("ask_last_question")

    if data in ("ask_more:quickqa", "ask_more:qna5"):  # accept both ids
        with contextlib.suppress(Exception):
            await q.message.edit_reply_markup(reply_markup=None)
        with contextlib.suppress(Exception):
            await q.message.chat.send_action(action=ChatAction.TYPING)

        ok, text = await _gen_quick_qna(subject=subject, concept=concept, n=5)

        # Chunk to fit Telegram limits
        for i in range(0, len(text), 3800):
            await update.effective_chat.send_message(
                text[i:i+3800],
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        return



async def ask_receive_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle plain-text question for Ask flow."""
    from telegram.constants import ChatAction
    import asyncio, contextlib, html

    # Small local helpers (no external dependencies)
    def _chunks(s: str, n: int = 3800):
        s = s or ""
        for i in range(0, len(s), n):
            yield s[i:i+n]

    # very light sanitizer → HTML-safe + normalize bullets/newlines
    def _clean_to_html(s: str) -> str:
        s = s or ""
        # If upstream already used <b>, <i>, etc., keep them; escape everything else safely
        # First, escape fully…
        esc = html.escape(s, quote=False)
        # …then revert our minimal allowlist tags if model produced them literally
        esc = esc.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
        esc = esc.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
        esc = esc.replace("&lt;code&gt;", "<code>").replace("&lt;/code&gt;", "</code>")
        # normalize bullets
        esc = esc.replace("&bull;", "•")
        # Telegram HTML supports plain newlines; no <br> needed
        return esc

    try:  # OUTER try
        q = (update.message.text or "").strip()
        if not q:
            await update.message.reply_text("Please type a question, or send a photo, or /cancel.")
            return ASK_WAIT

        subject = context.user_data.get("ask_subject")
        context.user_data["ask_last_question"] = q

        with contextlib.suppress(Exception):
            await update.message.chat.send_action(action=ChatAction.TYPING)
        working = await update.message.reply_text("Working on it… 🧠")

        # Call solver (guarded; never crash the handler)
        try:
            ok, res = await ask_openai_vision(q, image_path=None, subject_hint=subject)
        except asyncio.TimeoutError:
            ok, res = False, "The solver timed out. Please try again."
        except Exception as e:
            ok, res = False, f"Error contacting solver: {e}"

        context.user_data["ask_last_question"] = q
        context.user_data["ask_last_answer"]  = res if ok else ""
        
        text = _clean_to_html(res if ok else f"Error: {res}")

        # Try editing the “working…” message first; if that fails, fall back to sends
        try:
            if len(text) <= 3800:
                await working.edit_text(
                    text,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                    reply_markup=_ask_followup_markup() if ok else None,
                )
            else:
                parts = list(_chunks(text))
                await working.edit_text(parts[0], parse_mode="HTML", disable_web_page_preview=True)
                for p in parts[1:]:
                    await update.message.reply_text(p, parse_mode="HTML", disable_web_page_preview=True)
                if ok:
                    await update.message.reply_text("What next?", reply_markup=_ask_followup_markup())
        except Exception:
            for p in _chunks(text):
                await update.message.reply_text(p, parse_mode="HTML", disable_web_page_preview=True)
            if ok:
                await update.message.reply_text("What next?", reply_markup=_ask_followup_markup())

        return ConversationHandler.END

    finally:  # always release flow lock
        with contextlib.suppress(Exception):
            unlock_flow(context)



async def ask_receive_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo question for Ask flow."""
    from telegram.constants import ChatAction
    import asyncio, contextlib, os, time, html

    def _chunks(s: str, n: int = 3800):
        s = s or ""
        for i in range(0, len(s), n):
            yield s[i:i+n]

    def _clean_to_html(s: str) -> str:
        s = s or ""
        esc = html.escape(s, quote=False)
        esc = esc.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
        esc = esc.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
        esc = esc.replace("&lt;code&gt;", "<code>").replace("&lt;/code&gt;", "</code>")
        esc = esc.replace("&bull;", "•")
        return esc

    local_path = None
    try:  # OUTER try
        photos = update.message.photo
        caption = (update.message.caption or "").strip()
        if not photos:
            await update.message.reply_text("Couldn’t find the image. Try again or send as a document.")
            return ASK_WAIT

        with contextlib.suppress(Exception):
            await update.message.chat.send_action(action=ChatAction.UPLOAD_PHOTO)

        tgfile = await context.bot.get_file(photos[-1].file_id)

        os.makedirs("tmp", exist_ok=True)
        local_path = os.path.join("tmp", f"ask_{update.effective_user.id}_{int(time.time())}.jpg")
        try:
            await tgfile.download_to_drive(local_path)
        except Exception as e:
            await update.message.reply_text(f"Download failed: {e}. Please try again.")
            return ASK_WAIT

        subject = context.user_data.get("ask_subject")
        prompt = caption if caption else "Solve the attached NEET-style problem. If MCQ, analyze options."
        context.user_data["ask_last_question"] = caption or "[image question]"

        with contextlib.suppress(Exception):
            await update.message.chat.send_action(action=ChatAction.TYPING)
        working = await update.message.reply_text("Got the image. Solving… 📷🧮")

        # Call solver (guarded)
        try:
            ok, res = await ask_openai_vision(prompt, image_path=local_path, subject_hint=subject)
        except asyncio.TimeoutError:
            ok, res = False, "The solver timed out while processing the image. Please try again."
        except Exception as e:
            ok, res = False, f"Error contacting solver: {e}"
        
        context.user_data["ask_last_question"] = caption or "[image question]"
        context.user_data["ask_last_answer"]  = res if ok else ""
        
        text = _clean_to_html(res if ok else f"Error: {res}")

        try:
            if len(text) <= 3800:
                await working.edit_text(
                    text,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                    reply_markup=_ask_followup_markup() if ok else None,
                )
            else:
                parts = list(_chunks(text))
                await working.edit_text(parts[0], parse_mode="HTML", disable_web_page_preview=True)
                for p in parts[1:]:
                    await update.message.reply_text(p, parse_mode="HTML", disable_web_page_preview=True)
                if ok:
                    await update.message.reply_text("What next?", reply_markup=_ask_followup_markup())
        except Exception:
            for p in _chunks(text):
                await update.message.reply_text(p, parse_mode="HTML", disable_web_page_preview=True)
            if ok:
                await update.message.reply_text("What next?", reply_markup=_ask_followup_markup())

        return ConversationHandler.END

    finally:
        # clean up temp file + release lock
        with contextlib.suppress(Exception):
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
        with contextlib.suppress(Exception):
            unlock_flow(context)


async def ask_followup_handler(update: "Update", context: "ContextTypes.DEFAULT_TYPE"):
    """
    Handles:
      ask_more:similar | ask_more:explain | ask_more:flash | ask_more:quickqa | ask_more:qna5

    Prefers quiz context from context.user_data["ask_more_ctx"], then falls back
    to legacy context keys: ask_subject / ask_last_question / ask_last_answer.
    Output is plain paragraphs (no asterisks/bullets), HTML-escaped for Telegram.
    """
    # --- imports needed here to avoid NameError in some deploy flows ---
    import contextlib, html as _html, re
    from telegram.constants import ChatAction
    from telegram.error import BadRequest

    q = update.callback_query
    await q.answer()
    data = (q.data or "").strip()

    # Best-effort: remove inline keyboard (avoid 400 on double-edit)
    with contextlib.suppress(BadRequest, Exception):
        await q.edit_message_reply_markup(reply_markup=None)

    # ---- pull context (quiz-first, then legacy) ----
    ctx_more  = context.user_data.get("ask_more_ctx") or {}
    subject   = (
        ctx_more.get("subject")
        or context.user_data.get("ask_subject")
        or context.user_data.get("quiz_subject")
        or "NEET"
    )
    last_q    = (ctx_more.get("question") or context.user_data.get("ask_last_question") or "").strip()
    last_a    = (ctx_more.get("answer_text") or context.user_data.get("ask_last_answer") or "").strip()
    expl_hint = (ctx_more.get("explanation") or "").strip()

    if not last_q:
        await context.bot.send_message(
            chat_id=q.message.chat.id,
            text="I lost the last question’s context. Please use /ask again.",
            reply_to_message_id=q.message.message_id,
        )
        return

    # --- helpers ---
    def _tg_chunks(s: str, n: int = 3800):
        s = s or ""
        for i in range(0, len(s), n):
            yield s[i:i+n]

    def _plainify(txt: str) -> str:
        """Light cleanup to match 'AI notes' style: no asterisks/bullets, keep paragraphs."""
        if not txt:
            return ""
        # remove markdown bold/italics markers and common bullet glyphs
        txt = txt.replace("**", "").replace("__", "").replace("_", "")
        txt = txt.replace("•", "")
        # strip leading list markers like "* ", "- ", "1) ", "1. "
        txt = re.sub(r"(?m)^\s*([*+-]|\d+[.)])\s+", "", txt)
        # collapse >2 consecutive newlines to exactly 2 (tidy paragraphs)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        return txt.strip()

    mode = (data.split(":", 1)[1] if ":" in data else "explain").lower()

    # --- Prompt builder with strict formatting contract ---
    base_instructions = (
        "Write in plain paragraphs. No asterisks, no bullets, no markdown, no code fences. "
        "Keep it concise and readable like short notes."
    )

    if mode in ("quickqa", "qna5"):
        prompt = (
            f"{base_instructions}\n"
            f"Create five short NEET practice Q&A items based on the concept of the problem below. "
            f"Each item should be 1–2 lines: a question then the answer on the next line. "
            f"Separate items with a blank line. Do not number or bullet them.\n\n"
            f"Subject: {subject}\n\n"
            f"Problem:\n{last_q}\n"
            + (f"\nReference solution (if helpful):\n{last_a}\n" if last_a else "")
        )
    elif mode == "similar":
        prompt = (
            f"{base_instructions}\n"
            f"Create ONE NEET-style problem similar in concept and difficulty to the original. "
            f"Then provide a brief worked solution and final answer. Use 2–4 short paragraphs total.\n\n"
            f"Subject: {subject}\n\n"
            f"Original problem:\n{last_q}\n"
            + (f"\nReference/approach (optional):\n{last_a}\n" if last_a else "")
        )
    elif mode == "flash":
        prompt = (
            f"{base_instructions}\n"
            f"Create five concise flashcard pairs (Q then A on next line) from the core ideas in the problem. "
            f"No numbering or bullets. Separate pairs with a blank line.\n\n"
            f"Subject: {subject}\n\n"
            f"Problem:\n{last_q}"
        )
    else:  # "explain"
        prompt = (
            f"{base_instructions}\n"
            f"Explain the key concepts behind solving this problem, then give a compact step-by-step approach, "
            f"two tips, and a common pitfall—each as short paragraphs (no lists).\n\n"
            f"Subject: {subject}\n\n"
            f"Problem:\n{last_q}\n"
            + (f"\nAuthor notes (optional):\n{expl_hint}\n" if expl_hint else "")
            + (f"\nExisting solution (optional):\n{last_a}\n" if last_a else "")
        )

    # typing indicator
    with contextlib.suppress(Exception):
        await q.message.chat.send_action(action=ChatAction.TYPING)

    # --- llm call ---
    raw = await call_openai(prompt)  # your helper should return plain str
    cleaned = _plainify(raw or "Could not generate a response right now.")
    safe_html = _html.escape(cleaned)

    for part in _tg_chunks(safe_html, 3800):
        await context.bot.send_message(
            chat_id=q.message.chat.id,
            text=part,
            parse_mode="HTML",
            disable_web_page_preview=True,
            reply_to_message_id=q.message.message_id,
        )


# ========================= Predictor =========================
CATEGORY_OPTIONS = [["General", "OBC", "EWS", "SC", "ST"]]

def tri_inline(prefix: str) -> InlineKeyboardMarkup:
    """Inline keyboard with Yes/No/Any → callback_data like 'prefix:yes'."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Yes", callback_data=f"{prefix}:yes"),
         InlineKeyboardButton("No",  callback_data=f"{prefix}:no"),
         InlineKeyboardButton("Any", callback_data=f"{prefix}:any")]
    ])

# --- quota-aware closing-rank helpers ---
def _get_close_from_quota(per_quota_map: Dict[str, Dict[str, Dict[str, int]]],
                          quota: str,
                          college_key: str,
                          category: str) -> Optional[int]:
    rec = (per_quota_map.get(quota) or {}).get(college_key)
    if not rec:
        return None
    return _get_close_rank_from_rec(rec, category)

def _pick_any_quota_record(per_quota_map: Dict[str, Dict[str, Dict[str, int]]],
                           college_key: str,
                           category: str,
                           air: Optional[int]) -> Tuple[Optional[int], Optional[str]]:
    best_close = None
    best_q = None
    for q in _ALLOWED_ANY_QUOTAS:
        cr = _get_close_from_quota(per_quota_map, q, college_key, category)
        if isinstance(cr, int):
            if (air is None) or (air <= cr):
                if (best_close is None) or (cr > best_close):
                    best_close, best_q = cr, q
    return best_close, best_q

def get_closing_rank(*,
                     college_key: str,
                     round_key: str,
                     quota: str,
                     category: str,
                     air: Optional[int]) -> Tuple[Optional[int], Optional[str], str]:
    """
    Resolve closing rank with strict handling:
      - Quota=='Any' → best among _ALLOWED_ANY_QUOTAS only (no legacy).
      - Quota=='AIQ' → per_quota first, then legacy AIQ fallback allowed.
      - Quota in {'Deemed','Central'} → per_quota only (NO legacy fallback).
    """
    per_quota = (CUTOFFS_Q.get(round_key) or {})
    legacy    = (CUTOFFS.get(round_key)   or {})
    

    if quota == "Any":
        cr, q_used = _pick_any_quota_record(per_quota, college_key, category, air)
        if isinstance(cr, int):
            return cr, (q_used or "Any"), "per_quota_any"
        # last resort for 'Any': legacy AIQ only
        rec = legacy.get(college_key, {})
        cr = _get_close_rank_from_rec(rec, category)
        if isinstance(cr, int) and ((air is None) or (air <= cr)):
            return cr, "AIQ", "legacy_aiq_any"
        return None, None, "none"

    # exact requested quota first
    cr = _get_close_from_quota(per_quota, quota, college_key, category)
    if isinstance(cr, int) and ((air is None) or (air <= cr)):
        return cr, quota, "per_quota"

    # IMPORTANT: only AIQ may fall back to legacy AIQ
    if quota == "AIQ":
        rec = legacy.get(college_key, {})
        cr = _get_close_rank_from_rec(rec, category)
        if isinstance(cr, int) and ((air is None) or (air <= cr)):
            return cr, "AIQ", "legacy_aiq"

    # For Deemed/Central/NRI etc, do NOT fall back to AIQ
    return None, None, "none"

def canonical_quota_ui(text: str) -> str:
    t = _norm_hdr(text)
    if t in {"AIQ","ALL INDIA","ALL INDIA QUOTA"}: return "AIQ"
    if "DEEMED" in t or "PAID" in t or "MANAGEMENT" in t: return "Deemed"
    if "CENTRAL" in t or t == "CU": return "Central"
    #if "AIIMS" in t:   return "AIIMS"
    #if "JIPMER" in t:  return "JIPMER"
    #if "ESIC" in t:    return "ESIC"
    #if "AFMS" in t or "ARMED" in t: return "AFMS"
    #if "STATE" in t or t in {"SQ","IPU","IP"}: return "State"
    #if t in {"ANY","ALL"}: return "Any"
    return "AIQ"

def _quota_bucket_from_ui(v: str) -> str:
    t = str(v or "").strip().upper()
    if t in {"AIQ", "ALL INDIA"}: return "AIQ"
    if "CENTRAL" in t:            return "Central"
    if "DEEMED" in t or "PAID" in t or "MANAGEMENT" in t: return "Deemed"
    if t in {"ANY", "ALL"}:       return "Any"
    return "AIQ"

def quota_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [["AIQ", "Deemed", "Central"], ],
        one_time_keyboard=True,
        resize_keyboard=True,
    )

def _pick_category_key(v: str) -> str:
    return _canon_cat(v)

def canonical_category(cat: str) -> str:
    cat = (cat or "").strip().lower()
    if cat in {"ur", "gen", "general"}: return "General"
    if cat in {"obc"}: return "OBC"
    if cat in {"ews"}: return "EWS"
    if cat in {"sc"}:  return "SC"
    if cat in {"st"}:  return "ST"
    return cat.title()



def _code_key(x: Any) -> str:
    """Normalize a college code key like 'C0065' -> 'C0065' (upper + strip)."""
    s = str(x or "").strip().upper()
    return re.sub(r"\s+", "", s)

def _lookup_close_rank(lookup: dict, name_key: str, code_key: Optional[str], quota: str, cat: str) -> Optional[int]:
    """
    Try (code, quota, cat) first if code present, then (name_key, quota, cat).
    Returns int or None.
    """
    if code_key:
        v = lookup.get((code_key, quota, cat))
        if isinstance(v, int):
            return v
    v = lookup.get((name_key, quota, cat))
    return int(v) if isinstance(v, int) else None

def _pick_col(cols, *cands):
    """Case-insensitive, 'contains' tolerant pick of the first matching column name."""
    cols = [str(c) for c in cols]
    norm = {c.lower().strip(): c for c in cols}
    for want in cands:
        w = want.lower().strip()
        if w in norm:
            return norm[w]
        for k, original in norm.items():
            if w == k or w in k:
                return original
    return None




def _fmt_money(v):
    try:
        s = str(v).replace(",", "").strip()
        if not s:
            return "—"
        n = float(s)
        return f"₹{int(n):,}"
    except Exception:
        return "—"


def _row_brief(r: dict) -> str:
    name = r.get("college_name") or r.get("College Name") or "—"
    addr = r.get("address") or r.get("city") or r.get("state") or ""

    # build the identifiers list (don't show any of these in output)
    id_candidates = [
        r.get("college_code"), r.get("code"),
        r.get("college_id"), r.get("institute_code"),
        r.get("college_name") or r.get("College Name"),
    ]

    # user context (set by caller: _row_brief._user_ctx = user)
    user = getattr(_row_brief, "_user_ctx", {}) or {}
    round_ui = user.get("cutoff_round") or user.get("round") or "2025_R1"
    quota    = user.get("quota") or "AIQ"
    category = user.get("category") or "OP"

    df_lookup = r.get("_df_lookup")  # may be None; fine

    cr = _closing_rank_for_identifiers(
        [x for x in id_candidates if x],
        round_ui, quota, category,
        df_lookup=df_lookup,
        lookup_dict=CUTOFF_LOOKUP,
    )
    cr_txt  = f"Closing Rank {cr}" if cr not in (None, "", "—") else "Closing Rank —"

    fee     = r.get("total_fee") or r.get("Fee")
    fee_txt = f"Annual Fee {_fmt_money(fee)}"

    header = f"{name}, {addr}".strip().rstrip(",")
    return f"{header}\n{cr_txt}\n{fee_txt}"


def _resolve_college_name_from_row(r: dict | pd.Series, name_col: str | None) -> tuple[str | None, str | None]:
    """
    Returns (display_name, code_for_debug). Tries code, then id, then raw name.
    """
    def _nk(x):
        s = "" if x is None else str(x).strip().upper()
        return re.sub(r"[^A-Z0-9]+", "", s)

    code = _nk(r.get("college_code") or r.get("College Code"))
    cid  = _nk(r.get("college_id")   or r.get("College ID"))

    if code and code in COLLEGE_NAME_BY_CODE:
        return COLLEGE_NAME_BY_CODE[code], code
    if cid and cid in COLLEGE_NAME_BY_ID:
        return COLLEGE_NAME_BY_ID[cid], code or cid

    raw = (str(r.get(name_col)) or "").strip() if name_col else ""
    return (raw or None), code or cid or None


def _close_rank_tolerant(key: tuple[str,str,str], cutoff_lookup: dict, per_round: dict[str, dict] | None = None):
    """Try exact key first; if missing, search same college/quota across any category/round present."""
    if key in cutoff_lookup:
        return cutoff_lookup[key]

    # If your loader doesn’t embed per-round, we’ll just try same code with any cat
    code_or_id, quota, cat = key
    for (k_code, k_quota, k_cat), v in cutoff_lookup.items():
        if k_code == code_or_id and k_quota == quota:
            return v  # first seen for that college+quota

    return None

    cq     = _canon_quota(quota)
    cats   = _cat_aliases(cat)
    keys   = [k for k in [norm_key] + (alt_keys or []) if k]

    def _get(k, q, c):
        return cutoff_lookup.get((k, _canon_quota(q), _canon_cat(c)))

    # 1) exact quota + cat aliases
    for k in keys:
        for c in cats:
            v = _get(k, cq, c)
            if v is not None:
                return _safe_int(v)

    # 2) any quota, cat aliases
    quotas_all = {q for (_k, q, _c) in cutoff_lookup.keys() if _k in keys}
    for k in keys:
        for q in quotas_all:
            for c in cats:
                v = _get(k, q, c)
                if v is not None:
                    return _safe_int(v)

    # 3) exact quota, any cat
    cats_all = {c for (_k, _q, c) in cutoff_lookup.keys() if _k in keys}
    for k in keys:
        for c in cats_all:
            v = _get(k, cq, c)
            if v is not None:
                return _safe_int(v)

    # 4) any quota, any cat -> best available (min rank)
    candidates = []
    for k in keys:
        for (_k, q, c), v in cutoff_lookup.items():
            if _k == k and v is not None:
                iv = _safe_int(v)
                if isinstance(iv, int):
                    candidates.append(iv)
    if candidates:
        return min(candidates)

    return None


def shortlist_and_score(colleges_df: pd.DataFrame, user: dict, cutoff_lookup: dict) -> list[dict]:
    """
    Rows: (id, name, state, close_rank, category, quota, score, nirf_rank, total_fee)

    - Column picking is local-safe (no name collision with global _pick).
    - Closing rank resolution:
        1) get_closing_rank(...)  [uses CUTOFFS_Q / CUTOFFS if available]
        2) fallback to the provided flat cutoff_lookup (CUTOFF_LOOKUP) with tolerant matching
    - Filter by eligibility: include only if (AIR is None) or (AIR <= close_rank)
    - Sort: close_rank ASC, NIRF ASC, name
    """
    out: list[dict] = []
    if colleges_df is None or len(colleges_df) == 0:
        return out

    # ---- local helpers (NO collision with global ones) ----
    def _pick_col_local(cols, *cands):
        cols = [str(c) for c in cols]
        norm = {c.lower().strip(): c for c in cols}
        for want in cands:
            w = want.lower().strip()
            if w in norm:
                return norm[w]
            for k, original in norm.items():
                if w == k or w in k:
                    return original
        return None

    def _resolve_from_flat_lookup(keys: list[str], quota: str, cat: str) -> int | None:
        """Try exact (key, quota, cat); then same key with any quota only if quota=='Any'."""
        if not cutoff_lookup:
            return None
        q = _canon_quota(quota)
        cat_aliases = _cat_aliases(cat)

        # 1) exact quota + cat aliases
        for k in keys:
            for c in cat_aliases:
                v = cutoff_lookup.get((k, q, _canon_cat(c)))
                if isinstance(v, int):
                    return v

        # 2) Only if user picked Any: same key, any quota + cat aliases
        if q == "Any":
            for k in keys:
                for (kk, qq, cc), v in cutoff_lookup.items():
                    if kk == k and _canon_cat(cc) in cat_aliases and isinstance(v, int):
                        return v

        # 3) If still nothing: give up (no “any quota any cat” leak)
        return None

    # ---- columns ----
    cols = list(map(str, colleges_df.columns))
    name_col = _pick_col_local(cols, "College Name", "college_name", "name", "institute_name")
    state_col = _pick_col_local(cols, "state", "State")
    code_col  = _pick_col_local(cols, "college_code", "College Code", "code", "institute_code")
    id_col    = _pick_col_local(cols, "college_id", "College ID", "id")
    nirf_col  = _pick_col_local(cols, "nirf_rank_medical_latest", "NIRF", "nirf")
    fee_col   = _pick_col_local(cols, "total_fee", "Fee")

    # ---- user prefs ----
    quota_ui  = _canon_quota(user.get("quota") or user.get("pref_quota") or "AIQ")
    category  = _canon_cat(user.get("category"))
    air       = _safe_int(user.get("rank_air") or user.get("air"))
    round_key = user.get("cutoff_round") or user.get("round") or "2025_R1"

    for _, r in colleges_df.iterrows():
        code_key = _norm_key(r.get(code_col)) if code_col else ""
        id_key   = _norm_key(r.get(id_col))   if id_col   else ""
        raw_name = (str(r.get(name_col)) if name_col else "") or ""
        raw_name = raw_name.strip()

        display_name = (
            (COLLEGE_NAME_BY_CODE.get(code_key) if code_key else None)
            or (COLLEGE_NAME_BY_ID.get(id_key) if id_key else None)
            or raw_name
            or "Unknown college"
        )

        # Prefer code, then id, then normalized name
        college_key = code_key or id_key or _name_key(raw_name)

        # 1) canonical resolver (CUTOFFS_Q / CUTOFFS)
        close_rank, quota_used, src = get_closing_rank(
            college_key=college_key,
            round_key=round_key,
            quota=quota_ui,
            category=category,
            air=air
        )

        # 2) fallback to the flat lookup provided (CUTOFF_LOOKUP)
        if close_rank is None:
            keys = [k for k in (code_key, id_key, _name_key(raw_name)) if k]
            cr2 = _resolve_from_flat_lookup(keys, quota_ui, category)
            if isinstance(cr2, int):
                close_rank, quota_used, src = cr2, quota_ui, "flat_lookup"

        # eligibility
        if close_rank is None:
            continue
        if air is not None and air > close_rank:
            continue

        nirf_val  = _safe_int(r.get(nirf_col)) if nirf_col else None
        fee_val   = _safe_int(r.get(fee_col))  if fee_col  else None
        state_val = (str(r.get(state_col)).strip() if state_col else "—")

        out.append({
            "college_id":   (str(r.get(id_col)) if id_col else None),
            "college_code": (str(r.get(code_col)) if code_col else None),
            "college_name": display_name,
            "state":        state_val,
            "close_rank":   int(close_rank),
            "category":     category,
            "quota":        quota_used or quota_ui,
            "source":       src,
            "score":        None,
            "nirf_rank":    nirf_val,
            "total_fee":    fee_val,
        })

    # ------ ONLY CHANGE HERE: if no results and AIR was provided, return [] ------
    if not out:
        if air is not None:
            return []
        # metadata-only fallback (kept for when AIR not provided)
        tmp = []
        for _, r in colleges_df.iterrows():
            tmp.append({
                "college_id":   (str(r.get(id_col)) if id_col else None),
                "college_code": (str(r.get(code_col)) if code_col else None),
                "college_name": (str(r.get(name_col)).strip() if name_col else "Unknown college"),
                "state":        (str(r.get(state_col)).strip() if state_col else "—"),
                "close_rank":   None,
                "category":     category,
                "quota":        quota_ui,
                "source":       "fallback",
                "score":        None,
                "nirf_rank":    _safe_int(r.get(nirf_col)) if nirf_col else None,
                "total_fee":    _safe_int(r.get(fee_col)) if fee_col else None,
            })
        tmp.sort(key=lambda x: (
            x["nirf_rank"] if x["nirf_rank"] is not None else 10**9,
            x["college_name"] or ""
        ))
        return tmp[:30]

    out.sort(key=lambda x: (
        x["close_rank"],
        x["nirf_rank"] if x["nirf_rank"] is not None else 10**9,
        x["college_name"] or ""
    ))
    return out

# Final cutoffs we read from your “Cutoffs” sheet

async def cutdiag(update, context):
    user = context.user_data
    round_ui   = user.get("round")
    quota_ui   = user.get("quota")
    category_ui= user.get("category")
    air        = user.get("rank_air")

    bucket  = _quota_bucket_from_ui(quota_ui or "")
    cat_key = _pick_category_key(category_ui or "")
    rounds  = list(CUTOFFS_Q.keys())
    has_r   = round_ui in CUTOFFS_Q
    has_b   = has_r and (bucket in CUTOFFS_Q.get(round_ui, {}))
    cnt     = len((CUTOFFS_Q.get(round_ui, {}) or {}).get(bucket, {}) or {})

    msg = (
        f"Diag\n"
        f"• Round: {round_ui}  present={has_r}\n"
        f"• Quota UI: {quota_ui}  → bucket={bucket}  present_in_round={has_b}\n"
        f"• Category: {category_ui} → {cat_key}\n"
        f"• AIR: {air}\n"
        f"• Cutoff keys in that bucket: {cnt}\n"
        f"• Rounds loaded: {rounds[:6]}{' …' if len(rounds)>6 else ''}"
    )
    await update.effective_chat.send_message(msg)

async def predict_mockrank_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    blocked = _start_flow(context, "predict")
    if blocked and blocked != "predict":
        tgt = _target(update)
        if tgt:
            await tgt.reply_text(
                f"You're currently in *{blocked}* flow. Send /cancel to exit it first.",
                parse_mode="Markdown"
            )
        return ConversationHandler.END
    tgt = _target(update)
    await tgt.reply_text("Enter your *mock test All-India Rank* (integer):", parse_mode="Markdown")
    return ASK_MOCK_RANK

async def predict_mockrank_collect_rank(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip().replace(",", "")
    if not txt.isdigit():
        await update.message.reply_text("Please send a valid integer rank.")
        return ASK_MOCK_RANK
    context.user_data["mock_rank"] = int(txt)
    await update.message.reply_text("How many candidates appeared in that mock (total participants)?")
    return ASK_MOCK_SIZE

async def predict_mockrank_collect_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip().replace(",", "")
    if not txt.isdigit() or int(txt) < 1:
        await update.message.reply_text("Please send a valid total participants count (integer ≥ 1).")
        return ASK_MOCK_SIZE

    mock_rank = context.user_data.get("mock_rank")
    size = int(txt)
    # percentile among participants: higher is better
    # Percentile ~ (1 - (rank-1)/size) * 100
    pct = max(0.0, min(100.0, (1.0 - (max(1, mock_rank)-1) / max(1, size)) * 100.0))

    # Map percentile to NEET AIR using a simple population model
    # AIR ≈ round((100 - pct)/100 * NEET_CANDIDATE_POOL_DEFAULT)
    air_est = max(1, int(round((100.0 - pct) / 100.0 * NEET_CANDIDATE_POOL_DEFAULT)))

    context.user_data["rank_air"] = air_est

    kb = quota_keyboard()
    await update.message.reply_text(
        f"Estimated NEET AIR from mock percentile ≈ *{air_est}*.\n\nSelect your *quota*:",
        parse_mode="Markdown",
        reply_markup=kb,
    )
    return ASK_QUOTA


async def predict_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    blocked = _start_flow(context, "predict")
    if blocked and blocked != "predict":
        tgt = _target(update)
        if tgt:
            await tgt.reply_text(
                f"You're currently in *{blocked}* flow. Send /cancel to exit it first.",
                parse_mode="Markdown"
            )
        return ConversationHandler.END

    for k in ("r", "category", "weights", "require_pg_quota", "avoid_bond", "domicile_state", "quota", "rank_air"):
        context.user_data.pop(k, None)

    tgt = _target(update)
    await tgt.reply_text("Send your NEET All India Rank (AIR) as a number (e.g., 15234).",
                         reply_markup=ReplyKeyboardRemove())
    return ASK_AIR

    
# ========== DEBUG: show loaded cutoff record for a college ==========
async def debug_loaded_record(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /debug_loaded_record <college name> [round_key] [quota]
    Example:
      /debug_loaded_record AIIMS New Delhi 2025_R1 AIQ
    If round/quota omitted, uses ACTIVE_CUTOFF_ROUND_DEFAULT and AIQ.
    """
    try:
        args = context.args or []
        if not args:
            await update.message.reply_text(
                "Usage:\n/debug_loaded_record <college name> [round_key] [quota]\n"
                "Ex: /debug_loaded_record AIIMS New Delhi 2025_R1 AIQ"
            )
            return

        round_key = None
        quota_sel = None
        tail = [a for a in args[-2:]]
        for t in reversed(tail):
            u = t.upper()
            if (round_key is None) and (("R" in u and any(d.isdigit() for d in u)) or "STRAY" in u or "202" in u):
                round_key = u
            elif (quota_sel is None) and u in {"AIQ", "STATE", "ALL_INDIA", "CENTRAL", "MANAGEMENT", "DEEMED"}:
                quota_sel = u

        name_tokens = []
        for a in args:
            u = a.upper()
            if u == round_key or u == (quota_sel or "").upper():
                continue
            name_tokens.append(a)
        q = " ".join(name_tokens).strip()
        if not q:
            await update.message.reply_text("Please provide a college name.")
            return

        rk = (round_key or ACTIVE_CUTOFF_ROUND_DEFAULT).upper()
        quota_sel = (quota_sel or "AIQ").upper()

        cut_map = (CUTOFFS_Q.get(rk) or {}).get(quota_sel, {}) or (CUTOFFS.get(rk) or {})
        if not cut_map:
            await update.message.reply_text(f"No cutoff data in memory for {rk}/{quota_sel}.")
            return

        target = _norm_name_key(q)
        best_key, best_score = None, 0.0

        def toks(s: str):
            return re.findall(r"[A-Z]+", s.upper())

        tset = set(toks(target))
        for k in cut_map.keys():
            kset = set(toks(k))
            inter = len(tset & kset)
            union = len(tset | kset) or 1
            score = inter / union
            if target in k or k in target:
                score += 0.5
            if score > best_score:
                best_key, best_score = k, score

        if not best_key or best_score < 0.4:
            await update.message.reply_text("No close college key found in loaded map.")
            return

        rec = cut_map.get(best_key, {})
        lines = [f"🔎 Loaded record for *{best_key}*  (round {rk}, quota {quota_sel}):"]
        for cat in ["General", "EWS", "OBC", "SC", "ST", "General_PwD", "EWS_PwD", "OBC_PwD", "SC_PwD", "ST_PwD"]:
            if cat in rec:
                lines.append(f"  • {cat}: {rec[cat]}")
        if len(lines) == 1:
            lines.append("(no categories present)")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except Exception as e:
        log.exception("debug_loaded_record failed")
        await update.message.reply_text(f"Error: {e}")



async def cutoff_probe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = " ".join(context.args or [])
    if not q:
        await update.message.reply_text("Usage: /cutoff_probe <part of college name>")
        return
    rk = context.user_data.get("cutoff_round", ACTIVE_CUTOFF_ROUND_DEFAULT)
    quota_sel = (context.user_data.get("quota") or "AIQ").strip().title()

    cut_map = (CUTOFFS_Q.get(rk) or {}).get(quota_sel, {}) or (CUTOFFS.get(rk) or {})
    if not cut_map:
        await update.message.reply_text(f"No cutoff data in memory for {rk}/{quota_sel}.")
        return

    target = _norm_name_key(q)
    best_key, best_score = None, 0.0

    def toks(s: str): return re.findall(r"[A-Z]+", s.upper())

    tset = set(toks(target))
    for k in cut_map.keys():
        kset = set(toks(k))
        inter = len(tset & kset)
        union = len(tset | kset) or 1
        score = inter/union
        if target in k or k in target:
            score += 0.5
        if score > best_score:
            best_key, best_score = k, score

    if not best_key or best_score < 0.4:
        await update.message.reply_text("No close college key found in loaded map.")
        return

    cat = canonical_category(context.user_data.get("category", "General"))
    air = context.user_data.get("rank_air")
    air_int = air if isinstance(air, int) else None

    cr, quota_used, src = get_closing_rank(
        college_key=best_key,
        round_key=rk,
        quota=quota_sel,
        category=cat,
        air=air_int
    )

    lines = [
        f"🔎 Loaded record for *{best_key}*  (round {rk}, requested quota {quota_sel}):",
        f"  • Category: {cat}",
    ]

    if cr is not None:
        lines.append(f"  • Resolved close rank: *{cr}*  (quota used: *{quota_used}*, source: *{src}*)")
    else:
        lines.append("  • No closing rank found for this selection.")

    per_quota_map = (CUTOFFS_Q.get(rk) or {})
    any_shown = False
    for qname, qmap in sorted(per_quota_map.items()):
        rec_q = (qmap or {}).get(best_key)
        if rec_q:
            val = _get_close_rank_from_rec(rec_q, cat)
            if isinstance(val, int):
                lines.append(f"    - {qname}: {cat} = {val}")
                any_shown = True

    if not any_shown:
        legacy = (CUTOFFS.get(rk) or {}).get(best_key, {})
        if legacy:
            val = _get_close_rank_from_rec(legacy, cat)
            if isinstance(val, int):
                lines.append(f"    - (legacy AIQ projection): {cat} = {val}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def on_air(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip().replace(",", "")
    if not txt.isdigit():
        await update.message.reply_text("Please send a valid integer for AIR.")
        return ASK_AIR

    context.user_data["rank_air"] = int(txt)

    prof = get_user_profile(update)
    if prof.get("category") in {"General", "OBC", "EWS", "SC", "ST"}:
        context.user_data["category"] = prof["category"]
        cat_hint = f"\nUsing saved category: *{prof['category']}*"
    else:
        cat_hint = ""

    kb = quota_keyboard()
    await update.message.reply_text(
        "Select your admission *quota* for prediction:"
        "\n• *AIQ* = All India Quota (MCC)"
        "\n• *Deemed* = Deemed Universities"
        "\n• *Central* = Central Universities"
      #  "\n• *AIIMS/JIPMER/ESIC/AFMS* as applicable"
        f"{cat_hint}",
        parse_mode="Markdown",
        reply_markup=kb,
    )
    return ASK_QUOTA


async def on_quota(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = canonical_quota_ui(update.message.text or "")
    if not q:
        await update.message.reply_text("Pick a valid quota.", reply_markup=quota_keyboard())
        return ASK_QUOTA

    context.user_data["quota"] = q

    kb = ReplyKeyboardMarkup([["General", "OBC", "EWS", "SC", "ST"]],
                             one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Select your category:", reply_markup=kb)
    return ASK_CATEGORY

async def on_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cat = canonical_category(update.message.text or "")
    if cat not in {"General", "OBC", "EWS", "SC", "ST"}:
        kb = ReplyKeyboardMarkup([["General", "OBC", "EWS", "SC", "ST"]],
                                 one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Pick a valid category.", reply_markup=kb)
        return ASK_CATEGORY

    context.user_data["category"] = cat
    kb = ReplyKeyboardMarkup([["Skip"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text(
        "Type your *domicile state* (e.g., Delhi, Uttar Pradesh) or tap *Skip*.",
        parse_mode="Markdown", reply_markup=kb
    )
    return ASK_DOMICILE

async def on_domicile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip()
    if txt.lower() != "skip" and txt:
        context.user_data["domicile_state"] = txt
    else:
        prof = get_user_profile(update)
        context.user_data["domicile_state"] = prof.get("domicile_state")

    # Do NOT ask for PG/bond now; keep features inactive:
    context.user_data["require_pg_quota"] = None
    context.user_data["avoid_bond"] = None

    # Go straight to results (no scoring / no preference UI)
    return await _finish_predict_now(update, context)

async def on_pg_req_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    try:
        _, choice = (q.data or "").split(":")
    except ValueError:
        choice = "any"
    context.user_data["require_pg_quota"] = True if choice == "yes" else False if choice == "no" else None

    try:
        await q.edit_message_reply_markup(None)
    except Exception:
        pass
    await q.message.reply_text(
        "Do you want to *AVOID* service bond colleges?",
        parse_mode="Markdown",
        reply_markup=tri_inline("bond"),
    )
    return ASK_BOND_AVOID

async def on_bond_avoid_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    val = (q.data or "bond:any").split(":", 1)[-1]
    context.user_data["avoid_bond"] = {"yes": True, "no": False}.get(val, None)
    return ASK_PREF

async def on_pg_req(update: Update, context: ContextTypes.DEFAULT_TYPE):
    t = (update.message.text or "").strip().lower()
    context.user_data["require_pg_quota"] = True if t == "yes" else False if t == "no" else None

    await update.message.reply_text(
        "Do you want to *AVOID* service bond colleges?",
        parse_mode="Markdown",
        reply_markup=tri_inline("bond")
    )
    return ASK_BOND_AVOID

async def on_bond_avoid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    t = (update.message.text or "").strip().lower()
    context.user_data["avoid_bond"] = True if t == "yes" else False if t == "no" else None
    return ASK_PREF

def _pretty_website(url: str) -> str:
    if not url:
        return ""
    u = url.strip()
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = u.rstrip("/")
    return f"www.{u}" if not u.startswith(("www.",)) else u

async def _unknown_cb(update, context):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass

# ========================= Global /cancel & Errors =========================
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    unlock_flow(context)
    await update.message.reply_text("Cancelled. Back to menu.")
    await show_menu(update)
    return ConversationHandler.END


async def cancel_predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _end_flow(context, "predict")
    try:
        await update.message.reply_text("Prediction cancelled.", reply_markup=ReplyKeyboardRemove())
    except Exception:
        if update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text("Prediction cancelled.", reply_markup=ReplyKeyboardRemove())
    await show_menu(update)
    return ConversationHandler.END


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = None
        if update and hasattr(update, "effective_chat") and update.effective_chat:
            chat_id = update.effective_chat.id
        log.exception("Exception while handling an update", exc_info=context.error)
        if chat_id:
            await context.bot.send_message(chat_id=chat_id, text="Oops — something went wrong. Please try again.")
    except Exception:
        # Avoid error loops
        pass

# ====== Round helpers & admin commands ======
async def set_round(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = (update.message.text or "").split()
    if len(args) < 2:
        await update.message.reply_text("Use: /set_round 2025  or  /set_round 2024")
        return
    choice = args[1].strip().lower()
    if choice in {"2025", "r1", "2025_r1"}:
        context.user_data["cutoff_round"] = "2025_R1"
    elif choice in {"2024", "stray", "2024_stray"}:
        context.user_data["cutoff_round"] = "2024_Stray"
    else:
        await update.message.reply_text("Unknown option. Use 2025 or 2024.")
        return
    await update.message.reply_text(
        f"Cutoff round set to *{context.user_data.get('cutoff_round', ACTIVE_CUTOFF_ROUND_DEFAULT)}*.",
        parse_mode="Markdown"
    )

async def which_round(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rk = context.user_data.get("cutoff_round", ACTIVE_CUTOFF_ROUND_DEFAULT)
    await update.message.reply_text(f"Current cutoff round: *{rk}*", parse_mode="Markdown")


async def list_cutoff_sheets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(EXCEL_PATH):
        await update.message.reply_text(f"Excel not found at {EXCEL_PATH}")
        return
    try:
        xl = pd.ExcelFile(EXCEL_PATH)
    except Exception as e:
        await update.message.reply_text(f"Open Excel failed: {e}")
        return

    lines = ["*Sheets found*:"]
    for s in xl.sheet_names:
        lines.append(f"• {s}")
    txt = "\n".join(lines)
    await update.message.reply_text(txt if len(txt) < 3800 else txt[:3800], parse_mode="Markdown")


async def cutoff_headers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(EXCEL_PATH):
        await update.message.reply_text(f"Excel not found at {EXCEL_PATH}")
        return
    try:
        xl = pd.ExcelFile(EXCEL_PATH)
    except Exception as e:
        await update.message.reply_text(f"Open Excel failed: {e}")
        return

    rk = context.user_data.get("cutoff_round", ACTIVE_CUTOFF_ROUND_DEFAULT)
    chosen = choose_cutoff_sheet(xl, rk) or xl.sheet_names[0]

    try:
        df0 = pd.read_excel(xl, sheet_name=chosen, header=None)
    except Exception as e:
        await update.message.reply_text(f"Read failed on sheet '{chosen}': {e}")
        return

    hdr = _find_header_row(df0)
    try:
        df = pd.read_excel(xl, sheet_name=chosen, header=hdr).dropna(how="all")
    except Exception as e:
        await update.message.reply_text(f"Read with header failed on '{chosen}': {e}")
        return

    df.columns = [str(c).strip() for c in df.columns]
    cols = [str(c) for c in df.columns]

    name_col = _pick_col(cols, "College Name", "Institute Name", "College", "Institute")
    cat_col = _pick_col(cols, "Category", "Seat Category", "Cat")
    close_col = _pick_col(cols, "ClosingRank", "Closing Rank", "Close Rank", "AIR", "Closing AIR", "Closing")
    quota_col = _pick_col(cols, "Quota", "Allotment", "Allotment Category")

    is_long = bool(cat_col and close_col)

    lines = [
        f"Round {rk} → sheet '{chosen}'",
        f"Detected header row: {hdr}",
        "Columns:",
        ", ".join(cols),
        "",
        f"Name column: {name_col or 'NOT FOUND'}",
    ]

    if is_long:
        lines.append("Detected shape: LONG-FORM")
        try:
            cat_vals = (
                df[cat_col]
                .dropna()
                .astype(str)
                .str.strip()
                .value_counts()
                .head(12)
            )
            cat_preview = []
            for raw, cnt in cat_vals.items():
                canon = _canon_cat_from_value(raw) or "(unmapped)"
                cat_preview.append(f"  • {raw}  →  {canon}  ({cnt})")
            if cat_preview:
                lines.append("Sample Category values → canonical:")
                lines.extend(cat_preview)
        except Exception:
            pass

    payload = "\n".join(lines)
    if len(payload) > 3800:
        payload = payload[:3800]
    await update.message.reply_text(payload)


async def use_cutoff_sheet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rk = context.user_data.get("cutoff_round", ACTIVE_CUTOFF_ROUND_DEFAULT)

    parts = (update.message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        if not os.path.exists(EXCEL_PATH):
            await update.message.reply_text(f"Excel not found at {EXCEL_PATH}")
            return
        try:
            xl = pd.ExcelFile(EXCEL_PATH)
        except Exception as e:
            await update.message.reply_text(f"Could not open Excel: {e}")
            return
        names = "\n".join(f"• {s}" for s in xl.sheet_names)
        await update.message.reply_text(
            "Usage: /use_cutoff_sheet <exact sheet name>\n\nAvailable sheets:\n" + names
        )
        return

    sheet = parts[1].strip()
    try:
        xl = pd.ExcelFile(EXCEL_PATH)
    except Exception as e:
        await update.message.reply_text(f"Could not open Excel: {e}")
        return

    if sheet not in xl.sheet_names:
        matches = [s for s in xl.sheet_names if sheet.lower() in s.lower()]
        msg = f"Sheet '{sheet}' not found.\n\nAvailable:\n" + "\n".join(f"• {s}" for s in xl.sheet_names)
        if matches:
            msg += "\n\nClose matches:\n" + "\n".join(f"• {m}" for m in matches)
        await update.message.reply_text(msg)
        return

    CUTSHEET_OVERRIDE[rk] = sheet
    global CUTOFFS
    cnt = len(CUTOFFS.get(rk, {}))
    await update.message.reply_text(
        f"Set *{rk}* → *{sheet}*. Reloaded *{cnt}* cutoff rows.\nTry /predict.",
        parse_mode="Markdown",
    )


async def set_cutsheet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ...
    CUTSHEET_OVERRIDE[rk] = sheet
    # RELOAD right now, using the override:
    global CUTOFF_LOOKUP
    CUTOFF_LOOKUP = load_cutoff_lookup_from_excel(
        path=EXCEL_PATH,
        sheet="Cutoffs",        # or your chosen sheet
        round_tag="2025_R1",
        require_quota=None,
        require_course_contains="MBBS",
        require_category_set=("General","EWS","OBC","SC","ST"),
    ) or {}
    await update.message.reply_text(
        f"Override set. {rk} now using *{Cutoffs}*. "
        f"Cutoff entries loaded: *{len(CUTOFF_LOOKUP)}*. Try /predict.",
        parse_mode="Markdown",
    )

try:
        import pandas as pd

        def _flatten_cutoff_lookup_to_df(lookup: dict) -> pd.DataFrame:
            rows = []
            # expected shape: lookup[round_code][quota][category] -> list(dict)
            if isinstance(lookup, dict):
                for round_code, qmap in lookup.items():
                    if not isinstance(qmap, dict):
                        continue
                    for quota, cmap in qmap.items():
                        if not isinstance(cmap, dict):
                            continue
                        for category, items in cmap.items():
                            if isinstance(items, list):
                                for it in items:
                                    row = dict(it)
                                    row.setdefault("round_code", round_code)
                                    row.setdefault("quota", quota)
                                    row.setdefault("category", category)
                                    rows.append(row)
                # fallback: flat dict keyed by tuples {(code, quota, category): {...}}
                if not rows:
                    for key, val in lookup.items():
                        if isinstance(key, tuple) and isinstance(val, dict):
                            row = dict(val)
                            if len(key) >= 1:
                                row.setdefault("college_code", key[0])
                            if len(key) >= 2:
                                row.setdefault("quota", key[1])
                            if len(key) >= 3:
                                row.setdefault("category", key[2])
                            rows.append(row)
            return pd.DataFrame(rows)

        def build_cutoffs_df(lookup, colleges) -> pd.DataFrame:
            if isinstance(lookup, list):
                df_candidate = _normalize_cutoffs_df(pd.DataFrame(lookup))
            elif isinstance(lookup, dict):
                df_candidate = _normalize_cutoffs_df(_flatten_cutoff_lookup_to_df(lookup))
            elif isinstance(lookup, pd.DataFrame):
                df_candidate = _normalize_cutoffs_df(lookup)
            else:
                df_candidate = pd.DataFrame()

            if (
                not df_candidate.empty
                and isinstance(colleges, pd.DataFrame)
                and not colleges.empty
                and "college_code" in df_candidate.columns
            ):
                meta_cols = [c for c in ["college_code","college_name","state","total_fee","hostel_available"]
                             if c in colleges.columns]
                if meta_cols:
                    df_candidate = df_candidate.merge(
                        colleges[meta_cols].drop_duplicates("college_code"),
                        on="college_code",
                        how="left",
                    )
            return df_candidate

        def _normalize_cutoffs_df(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()

            # Course
            if "Course" not in df.columns:
                df["Course"] = df["course"] if "course" in df.columns else "MBBS"
            df["Course"] = df["Course"].astype(str).str.upper()

            # ClosingRank normalization
            for col in ("ClosingRank", "closing", "closing_rank", "rank"):
                if col in df.columns:
                    df["ClosingRank"] = pd.to_numeric(df[col], errors="coerce")
                    break

            # Quota / Category normalization
            

            def _norm_category(c: str | None) -> str | None:
                if not c: return None
                c = str(c).strip()
                return {"General": "OP", "Open": "OP", "UR": "OP"}.get(c, c)

            if "quota" in df.columns:
                df["quota"] = df["quota"].map(_norm_quota)
            if "category" in df.columns:
                df["category"] = df["category"].map(_norm_category)

            # round/code
            if "round_code" not in df.columns:
                df["round_code"] = df.get("round", "2025_R1")
            if "college_code" not in df.columns and "code" in df.columns:
                df["college_code"] = df["code"]

            return df

except Exception:
    log.exception("[startup] unexpected error in previous try-block")       

async def _finish_predict_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Build shortlist with current context (no scoring, no extra prompts).
    Sends ONE compact message + an 'AI Notes' button.
    Also: if no results under the chosen quota, suggest Deemed colleges (fee low → high).
    """
    # ---------- small local helpers (self-contained) ----------
    
    
    def _as_records(df_or_list):
        if hasattr(df_or_list, "to_dict"):
            return df_or_list.to_dict("records")
        return list(df_or_list or [])

    def _is_missing(v):
        try:
            s = str(v).strip().lower()
        except Exception:
            return True
        return s in {"", "—", "na", "n/a", "nan", "none"}

    def _safe_str(v, default: str = "") -> str:
        try:
            if v is None: return default
            if isinstance(v, float) and math.isnan(v): return default
            s = str(v).strip()
            return default if s.lower() in NA_STRINGS else s
        except Exception:
            return default

    def _num_or_inf(v):
        try:
            s = str(v).replace(",", "").strip()
            if not s or s.lower() in {"na", "n/a", "nan"}:
                return float("inf")
            return float(s)
        except Exception:
            return float("inf")

    def _fmt_rank_val(v):
        try:
            if _is_missing(v):
                return "—"
            return f"{int(float(str(v))):,}"
        except Exception:
            return "—"

    def _fmt_money(v):
        try:
            if _is_missing(v):
                return "—"
            n = float(str(v).replace(",", "").strip())
            return f"₹{int(n):,}"
        except Exception:
            return "—"

    def _pick(d: dict, *keys):
        for k in keys:
            val = d.get(k)
            if not _is_missing(val):
                return val
        return None

    

    def _deemed_only(rows):
        out = []
        for r in rows:
            own = _safe_str(r.get("ownership")).lower()
            if "deemed" in own:  # strict deemed filter
                out.append(r)
        return out

    def _sorted_deemed_by_fee(colleges, limit=10):
        rows = _as_records(colleges)
        rows = _deemed_only(rows)
        # keep MBBS if other courses exist in the same sheet
        rows = [r for r in rows if _safe_str(r.get("course") or "MBBS").upper() == "MBBS"]
        rows.sort(key=lambda x: _num_or_inf(x.get("total_fee") or x.get("Fee")))
        return rows[:limit]
    # -----------------------------------------------------------

    try:
        user = context.user_data or {}
        log.info(
            "_finish_predict_now user context: quota=%s, category=%s, air=%s",
            user.get("quota"), user.get("category"),
            user.get("rank_air") or user.get("air")
        )
        chat_id = update.effective_chat.id

        # don’t restrict by PG/bond here
        user["require_pg_quota"] = None
        user["avoid_bond"] = None

        # shortlist
        results = shortlist_and_score(COLLEGES, user, cutoff_lookup=CUTOFF_LOOKUP) or []
        results = _dedupe_results(results)[:15]  # show up to 15

        round_ui = user.get("cutoff_round") or user.get("round") or "2025_R1"
        header_plain = f"🔎 Using cutoff round: {round_ui}"

        df_lookup = context.application.bot_data.get("CUTOFFS_DF", None)

        # ---------- No results? Show Deemed (fee low → high) ----------
        if not results:
            deemed = _sorted_deemed_by_fee(COLLEGES, limit=10)

            if deemed:
                # allow formatter to resolve ranks if needed
                for d in deemed:
                    d["_df_lookup"] = df_lookup

                # Remember for AI notes
                context.user_data["LAST_SHORTLIST"] = deemed

                body = "\n\n".join(
                    f"{i}. {_format_row_multiline(r, user, df_lookup)}"
                    for i, r in enumerate(deemed, 1)
                )

                quota = (user or {}).get("quota") or "AIQ"
                category = (user or {}).get("category") or "General"

                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"{header_plain}\n\n"
                        f"Couldn’t find matches under {quota} / {category}.\n"
                        f"Here are Deemed colleges (sorted by lowest fee):\n\n{body}"
                    ),
                )

                # AI Notes button (keep it in predict, not in main menu)
                kb = InlineKeyboardMarkup(
                    [[InlineKeyboardButton("🧠 Get AI notes on these colleges", callback_data="ai_notes")]]
                )
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="Want quick expert-style notes on this shortlist?",
                    reply_markup=kb,
                )

                _end_flow(context, "predict")
                return ConversationHandler.END

            # If even deemed list is empty
            await context.bot.send_message(chat_id=chat_id, text=f"{header_plain}\n\nNo colleges found.")
            _end_flow(context, "predict")
            return ConversationHandler.END

        # ---------- We have results: show compact top-10 ----------
        top = results[:10]
        # remember these for AI notes (use the actual rows, not just the display compact)
        context.user_data["LAST_SHORTLIST"] = top

        # let the formatter resolve ranks via df lookup
        for r in top:
            try:
                r["_df_lookup"] = df_lookup
            except Exception:
                pass

        body = "\n\n".join(
            f"{i}. {_format_row_multiline(r, user, df_lookup)}"
            for i, r in enumerate(top, 1)
        )
        await context.bot.send_message(chat_id=chat_id, text=f"{header_plain}\n\n{body}")

        # AI Notes button
        kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🧠 Get AI notes on these colleges", callback_data="ai_notes")]]
        )
        await context.bot.send_message(
            chat_id=chat_id,
            text="Want quick expert-style notes on this shortlist?",
            reply_markup=kb,
        )

        _end_flow(context, "predict")
        return ConversationHandler.END

    except Exception as e:
        log.exception("_finish_predict_now failed", exc_info=e)
        try:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="❌ Error preparing shortlist.")
        except Exception:
            pass
        _end_flow(context, "predict")
        return ConversationHandler.END

async def on_pref(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles ASK_PREF: user just chose their preference profile.
    Builds shortlist using COLLEGES and CUTOFF_LOOKUP, then sends top results.
    """
    try:
        pref_label = (update.message.text or "").strip()
        context.user_data["pref_type"] = pref_label

        user = context.user_data or {}
        log.info("on_pref user context: quota=%s, category=%s, air=%s",
            user.get("quota"), user.get("category"), user.get("rank_air") or user.get("air"))
        chat_id = update.effective_chat.id

        results = shortlist_and_score(COLLEGES, user, cutoff_lookup=CUTOFF_LOOKUP) or []
        results = _dedupe_results(results)
        results = results[:15]

        round_ui = user.get("cutoff_round") or user.get("round") or "2025_R1"
        header_plain = f"🔎 Using cutoff round: {round_ui}"

        if not results:
            await context.bot.send_message(chat_id=chat_id, text=f"{header_plain}\n\nNo colleges found.")
            return ConversationHandler.END

        
        _end_flow(context, "predict")
        await show_menu(update)
        return ConversationHandler.END

    except Exception as e:
        log.exception("on_pref failed", exc_info=e)
        try:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="❌ Error preparing shortlist.")
        except Exception:
            pass
        _end_flow(context, "predict")
        return ConversationHandler.END

# ========================= Main =========================
def main():
    global CUTOFF_LOOKUP, COLLEGES, COLLEGE_META_INDEX

    # --- Build the Telegram app ---
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # --- Load datasets (robust, with logs) ---
    if not os.path.exists(EXCEL_PATH):
        log.error("Excel not found at %s", EXCEL_PATH)
        raise FileNotFoundError(EXCEL_PATH)

    # 1) Load Colleges (robust)
    COLLEGES = load_colleges_dataset(EXCEL_PATH)  # <- returns DataFrame (maybe minimal), never None
    if COLLEGES is None:
        COLLEGES = pd.DataFrame()
    if COLLEGES.empty:
        log.warning("load_colleges_dataset returned empty; continuing with empty DataFrame")
    else:
        log.info("Loaded colleges: %d rows, columns=%s", len(COLLEGES), list(COLLEGES.columns))

    # 2) Build name maps from the colleges DF you just loaded
    try:
        build_name_maps_from_colleges_df(COLLEGES)
    except Exception:
        log.exception("Failed building name maps from Colleges DF")

    # 3) Build/Load cutoffs lookup (your existing function)
    try:
        CUTOFF_LOOKUP = load_cutoff_lookup_from_excel(
            path=EXCEL_PATH,
            sheet="Cutoffs",
            round_tag="2025_R1",
            require_quota=None,
            require_course_contains="MBBS",
            require_category_set=("General","EWS","OBC","SC","ST"),
        ) or {}
    except Exception:
        log.exception("Failed to load cutoff lookup; continuing with empty")
        CUTOFF_LOOKUP = {}

    # 3b) Build normalized CUTOFFS_DF exactly once from the lookup
    try:
        CUTOFFS_DF = build_cutoffs_df(CUTOFF_LOOKUP, COLLEGES)  # << your helper (keep only one definition of it)
        globals()["CUTOFFS_DF"] = CUTOFFS_DF
        application.bot_data["CUTOFFS_DF"] = CUTOFFS_DF
        log.info("[startup] CUTOFFS_DF ready: %s rows", len(CUTOFFS_DF))

        # Optional smoke test (helpful while debugging)
        if not CUTOFFS_DF.empty and _has("_filter_predict"):
            try:
                _smoke = _filter_predict(
                    CUTOFFS_DF,
                    rank_air=1, quota="AIQ", domicile_state=None, category="OP",
                    course="MBBS", year=2025, round_code="2025_R1", limit=3, pwd_filter="N",
                )
                log.info("[startup] AIQ smoke rows=%d head=%s",
                         len(_smoke),
                         (_smoke[["college_code","ClosingRank"]].head(3).to_dict("records") if not _smoke.empty else []))
            except Exception:
                log.exception("[startup] AIQ smoke failed")
    except Exception:
        log.exception("[startup] Failed to prepare CUTOFFS_DF")
        CUTOFFS_DF = pd.DataFrame()
        application.bot_data["CUTOFFS_DF"] = CUTOFFS_DF

    log.info(
        "Starting bot… colleges: %d | cutoff entries: %d",
        len(COLLEGES),
        len(CUTOFF_LOOKUP),
    )

    # 4) Startup banner
    col_count = (0 if not isinstance(COLLEGES, pd.DataFrame) or COLLEGES.empty else len(COLLEGES))
    cut_count = (len(CUTOFF_LOOKUP) if isinstance(CUTOFF_LOOKUP, dict) else 0)
    round_code = globals().get("ACTIVE_CUTOFF_ROUND_DEFAULT")

    if round_code:
        log.info("Starting bot… colleges: %d | cutoff entries: %d (round=%s)", col_count, cut_count, round_code)
    else:
        log.info("Starting bot… colleges: %d | cutoff entries: %d", col_count, cut_count)

    # --- Helpers to add handlers only if callbacks exist (prevents NameError) ---
    def _has(*names: str) -> bool:
        g = globals()
        return all((n in g and callable(g[n])) for n in names)

   
    
    

    # --- Error handler ---
    if _has("on_error"):
        application.add_error_handler(on_error)

    # --- Basic commands ---
    if _has("start"):
        _add(CommandHandler("start", start), group=0)
        _add(CommandHandler("menu", start), group=0)
    if _has("reset_lock"):
        _add(CommandHandler("reset", reset_lock), group=0)
    if _has("quizdiag"):    
        _add(CommandHandler("quizdiag", quizdiag), group=0)

def _resolve_excel_path() -> str:  
    
    # --- Admin commands ---
    if _has("set_round"):           _add(CommandHandler("set_round", set_round), group=5)
    if _has("which_round"):         _add(CommandHandler("which_round", which_round), group=5)
    if _has("list_cutoff_sheets"):  _add(CommandHandler("list_sheets", list_cutoff_sheets), group=5)
    if _has("use_cutoff_sheet"):    _add(CommandHandler("use_cutoff_sheet", use_cutoff_sheet), group=5)
    if _has("cutoff_headers"):      _add(CommandHandler("cutoff_headers", cutoff_headers), group=5)
    if _has("set_cutsheet"):        _add(CommandHandler("set_cutsheet", set_cutsheet), group=5)
    if _has("cutoff_probe"):        _add(CommandHandler("cutoff_probe", cutoff_probe), group=5)
    if _has("cutdiag"):             _add(CommandHandler("cutdiag", cutdiag), group=5)
    if _has("quota_counts"):        _add(CommandHandler("quota_counts", quota_counts), group=5)

    # --- Ask (Doubt) conversation ---
    if _has("ask_start", "ask_subject_select", "ask_receive_photo", "ask_receive_text", "cancel"):
        ask_conv = ConversationHandler(
            entry_points=[
                CommandHandler("ask", ask_start),
                CallbackQueryHandler(ask_start, pattern=r"^menu_ask$"),
            ],
            states={
                ASK_SUBJECT: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_subject_select)],
                ASK_WAIT: [
                    MessageHandler(filters.PHOTO, ask_receive_photo),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, ask_receive_text),
                ],
            },
            fallbacks=[CommandHandler("cancel", cancel)],
            name="ask_conv",
            persistent=False,
            per_message=False,
        )
        _add(ask_conv, group=1)

    # --- Quiz conversation ---
    if _has("quiz_start", "quiz_subject", "quiz_difficulty", "quiz_size", "cancel"):
        quiz_conv = ConversationHandler(
            entry_points=[
                CommandHandler("quiz", quiz_start),
                CallbackQueryHandler(quiz_start, pattern=r"^menu_quiz$"),
            ],
            states={
                QUIZ_SUBJECT:    [MessageHandler(filters.TEXT & ~filters.COMMAND, quiz_subject)],
                QUIZ_DIFFICULTY: [MessageHandler(filters.TEXT & ~filters.COMMAND, quiz_difficulty)],
                QUIZ_SIZE:       [MessageHandler(filters.TEXT & ~filters.COMMAND, quiz_size)],
                QUIZ_RUNNING:    [CallbackQueryHandler(quiz_answer, pattern=r"^QUIZ:")],
            },
            fallbacks=[CommandHandler("cancel", cancel)],
            name="quiz_conv",
            persistent=False,
            per_message=False,
        )
        _add(quiz_conv, group=2)
        if _has("quiz_answer"):
            _add(CallbackQueryHandler(quiz_answer, pattern=r"^QUIZ:"), group=2)
        if _has("quiz_review_choice"):
            _add(CallbackQueryHandler(quiz_review_choice, pattern=r"^QUIZ_REVIEW:(yes|no)$"), group=2)
        if _has("quiz_predict_choice_noop"):
            _add(CallbackQueryHandler(quiz_predict_choice_noop, pattern=r"^QUIZ_PREDICT:no$"), group=2)

    # --- Predictor conversation ---
    if _has("predict_start", "on_air", "on_quota", "on_category",
            "on_domicile", "on_pg_req_cb", "on_pg_req", "on_bond_avoid_cb", "on_bond_avoid",
            "on_pref", "cancel_predict"):
        predict_conv = ConversationHandler(
            entry_points=[
                CommandHandler("predict", predict_start),
                CallbackQueryHandler(predict_start, pattern=r"^menu_predict$"),
                CommandHandler("mockpredict", predict_mockrank_start),
                CallbackQueryHandler(predict_mockrank_start, pattern=r"^menu_predict_mock$"),
            ],
            states={
                ASK_AIR:        [MessageHandler(filters.TEXT & ~filters.COMMAND, on_air)],
                ASK_MOCK_RANK:  [MessageHandler(filters.TEXT & ~filters.COMMAND, predict_mockrank_collect_rank)],
                ASK_MOCK_SIZE:  [MessageHandler(filters.TEXT & ~filters.COMMAND, predict_mockrank_collect_size)],
                ASK_QUOTA:      [MessageHandler(filters.TEXT & ~filters.COMMAND, on_quota)],
                ASK_CATEGORY:   [MessageHandler(filters.TEXT & ~filters.COMMAND, on_category)],
                ASK_DOMICILE:   [MessageHandler(filters.TEXT & ~filters.COMMAND, on_domicile)],
            },
            fallbacks=[CommandHandler("cancel", cancel_predict)],
            name="predict_conv",
            persistent=False,
            per_message=False,
        )
        _add(predict_conv, group=3)
    
    # --- Profile conversation ---
    if _has("setup_profile", "profile_menu", "profile_set_category", "profile_set_domicile",
            "profile_set_pref", "profile_set_email", "profile_set_mobile", "profile_set_primary", "cancel"):
        profile_conv = ConversationHandler(
            entry_points=[
                CommandHandler("profile", setup_profile),
                CallbackQueryHandler(setup_profile, pattern=r"^menu_profile$"),
            ],
            states={
                PROFILE_MENU:         [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_menu)],
                PROFILE_SET_CATEGORY: [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_set_category)],
                PROFILE_SET_DOMICILE: [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_set_domicile)],
                PROFILE_SET_PREF:     [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_set_pref)],
                PROFILE_SET_EMAIL:    [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_set_email)],
                PROFILE_SET_MOBILE: [
                    MessageHandler(filters.CONTACT, profile_set_mobile),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, profile_set_mobile),
                ],
                PROFILE_SET_PRIMARY:  [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_set_primary)],
            },
            fallbacks=[CommandHandler("cancel", cancel)],
            name="profile_conv",
            persistent=False,
        )
        _add(profile_conv, group=4)

    # --- AI Coach (Preference-List) ---
    if _has("coach_start", "coach_adjust_cb", "coach_save_cb"):
        _add(CommandHandler("coach", coach_start), group=0)
        # If you added a menu button with callback_data="menu_coach", enable this:
        _add(CallbackQueryHandler(coach_start, pattern=r"^menu_coach$"), group=0)
    
    if _has("coach_notes_cb"):
        _add(CallbackQueryHandler(coach_notes_cb, pattern=r"^coach_notes:v1$"), group=0)

    if _has("ai_notes_from_shortlist"):
        _add(CallbackQueryHandler(ai_notes_from_shortlist, pattern=r"^ai_notes$"), group=0)


    # --- Unknown callbacks last (safety net) ---
    if _has("_unknown_cb"):
        _add(CallbackQueryHandler(_unknown_cb), group=9)

    # --- Run bot (call ONCE) ---
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

# === BEGIN PATCH ===
import os, json, time, logging, pandas as pd, re
from pathlib import Path
from typing import Dict, List, Any, Optional

from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ConversationHandler, filters
)

log = logging.getLogger("aceit-bot")

# Globals your old code used
CUTOFF_LOOKUP: Dict[str, Any] = {}
COLLEGES: pd.DataFrame = pd.DataFrame()
COLLEGE_META_INDEX: Dict[str, Any] = {}
CUTOFFS_DF: pd.DataFrame = pd.DataFrame()

# --- Paths & config ---
REPO_DIR = Path(__file__).parent
DEFAULT_FILENAME = "MCC_Final_with_Cutoffs_2024_2025.xlsx"
DATA_DIR = REPO_DIR / "data"

# Prefer EXCEL_PATH env; else default to the repo-root filename above
EXCEL_PATH = os.getenv("EXCEL_PATH", str(REPO_DIR / DEFAULT_FILENAME))
# Optional: direct download URL if the file isn't in the repo
EXCEL_URL  = os.getenv("EXCEL_URL", "")






def _safe_df(v) -> pd.DataFrame:
    """Return a DataFrame for any input without triggering pandas truthiness errors."""
    if isinstance(v, pd.DataFrame):
        return v
    if v is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(v)
    except Exception:
        return pd.DataFrame()


def _resolve_excel_path() -> str:
    """
    Find (or download) the Excel file in common locations.
    Never registers handlers here. Only resolves and logs.
    """
    candidates = [
        Path(EXCEL_PATH),
        REPO_DIR / DEFAULT_FILENAME,
        DATA_DIR / DEFAULT_FILENAME,
        DATA_DIR / "aceit.xlsx",  # legacy fallback
    ]
    for p in candidates:
        if p.exists():
            log.info("Excel found at %s", p)
            return str(p)

    if EXCEL_URL:
        try:
            import requests
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            dest = DATA_DIR / DEFAULT_FILENAME
            log.info("Downloading Excel from EXCEL_URL …")
            r = requests.get(EXCEL_URL, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            log.info("Downloaded Excel to %s (%d bytes)", dest, dest.stat().st_size)
            return str(dest)
        except Exception:
            log.exception("Failed to download Excel from EXCEL_URL")

    log.error("Excel not found. Looked in: %s", [str(x) for x in candidates])
    # Return the first candidate path so downstream errors mention a concrete path
    return str(candidates[0])



    

# ---------- 1) STARTUP: load datasets once ----------
async def on_startup(app: Application):
    """
    Runs once when the FastAPI app starts.
    Loads your Excel, builds lookups/DFs, and stores CUTOFFS_DF in app.bot_data.
    Uses ACTIVE_CUTOFF_ROUND_DEFAULT.
    """
    global CUTOFF_LOOKUP, COLLEGES, COLLEGE_META_INDEX, CUTOFFS_DF

    # Resolve Excel
    excel_file = _resolve_excel_path()

    # 1) Load colleges
    try:
        df = load_colleges_dataset(excel_file)  # provided elsewhere in your code
    except Exception:
        log.exception("load_colleges_dataset failed")
        df = None
    COLLEGES = _safe_df(df)
    if COLLEGES.empty:
        log.warning("Loaded colleges is empty")
    else:
        log.info("Loaded colleges: %d rows, columns=%s", len(COLLEGES), list(COLLEGES.columns))

    # 2) Build name maps
    try:
        build_name_maps_from_colleges_df(COLLEGES)  # provided elsewhere
    except Exception:
        log.exception("Failed building name maps from Colleges DF")

    # 3) Build/Load cutoffs lookup (use your DEFAULT round)
    try:
        CUTOFF_LOOKUP = load_cutoff_lookup_from_excel(
            path=excel_file,
            sheet="Cutoffs",
            round_tag=ACTIVE_CUTOFF_ROUND_DEFAULT,   # provided elsewhere
            require_quota=None,
            require_course_contains="MBBS",
            require_category_set=("General", "EWS", "OBC", "SC", "ST"),
        )
        if not isinstance(CUTOFF_LOOKUP, dict):
            CUTOFF_LOOKUP = dict(CUTOFF_LOOKUP or {})
    except Exception:
        log.exception("Failed to load cutoff lookup; continuing with empty")
        CUTOFF_LOOKUP = {}

    # 3b) Normalize into CUTOFFS_DF once
    try:
        cuts_df = build_cutoffs_df(CUTOFF_LOOKUP, COLLEGES)  # provided elsewhere
    except Exception:
        log.exception("[startup] Failed to prepare CUTOFFS_DF")
        cuts_df = None
    CUTOFFS_DF = _safe_df(cuts_df)
    app.bot_data["CUTOFFS_DF"] = CUTOFFS_DF
    log.info("[startup] CUTOFFS_DF ready: %d rows", len(CUTOFFS_DF))



    # One consolidated banner
    col_count = (0 if not isinstance(COLLEGES, pd.DataFrame) or COLLEGES.empty else len(COLLEGES))
    cut_count = (len(CUTOFF_LOOKUP) if isinstance(CUTOFF_LOOKUP, dict) else 0)
    round_code = globals().get("ACTIVE_CUTOFF_ROUND_DEFAULT")

    if round_code:
        log.info("Starting bot… colleges: %d | cutoff entries: %d (round=%s)", col_count, cut_count, round_code)
    else:
        log.info("Starting bot… colleges: %d | cutoff entries: %d", col_count, cut_count)


def _has(*names: str) -> bool:
    """Return True only if all given names exist and are callable."""
    g = globals()
    return all((n in g and callable(g[n])) for n in names)

def _has_all(*names: str) -> bool:
    """Return True only if all given names exist (useful for state constants)."""
    g = globals()
    return all(n in g for n in names)


# ---------- 2) WIRING: attach handlers to existing Application ----------

from telegram.ext import CommandHandler, CallbackQueryHandler  # make sure this import is present

def register_handlers(app: Application) -> None:
    """
    Wire up all handlers safely. Uses _has/_has_all gates if present;
    otherwise treats symbols as present.
    """

    # --- local helper to keep calls tidy ---
    def _add(h, group: int = 0) -> None:
        app.add_handler(h, group=group)

    # --- guard helpers (fallbacks if you don't already have them) ---
    try:
        _has        # type: ignore[name-defined]
    except NameError:
        def _has(*names: str) -> bool:  # fallback: assume things exist
            return True
    try:
        _has_all    # type: ignore[name-defined]
    except NameError:
        def _has_all(*names: str) -> bool:  # fallback: assume things exist
            return True

    # ===================== Error handler =====================
    if _has("on_error"):
        app.add_error_handler(on_error)

    # ===================== Basic commands =====================
    if _has("start"):
        _add(CommandHandler("start", start), group=0)
        # Prefer /menu → show_menu (not alias to start)
        if _has("show_menu"):
            _add(CommandHandler("menu", show_menu), group=0)
        else:
            _add(CommandHandler("menu", start), group=0)

    if _has("reset_lock"):
        _add(CommandHandler("reset", reset_lock), group=0)

    # Diagnostics (optional)
    if _has("quizdiag"):
        _add(CommandHandler("quizdiag", quizdiag), group=0)

    # ===================== QUIZ shortcuts =====================
    if _has("quiz5"):         _add(CommandHandler("quiz5", quiz5), group=0)
    if _has("quiz10"):        _add(CommandHandler("quiz10", quiz10), group=0)
    if _has("quiz10physics"): _add(CommandHandler("quiz10physics", quiz10physics), group=0)
    if _has("quiz5medium"):   _add(CommandHandler("quiz5medium", quiz5medium), group=0)

    # ===================== QUIZ menu & router =====================
    # top-level "Quiz" button in main menu
    if _has("menu_quiz_handler"):
        _add(CallbackQueryHandler(menu_quiz_handler, pattern=r"^menu_quiz$"), group=0)

    # unified router for quiz picker + back
    if _has("quiz_menu_router"):
        _add(
            CallbackQueryHandler(
                quiz_menu_router,
                pattern=r"^(quiz:(mini5|mini10|sub:.+|streaks|leaderboard)|menu:back)$",
            ),
            group=0,
        )

    # answer buttons from running quiz
    if _has("on_answer"):
        _add(CallbackQueryHandler(on_answer, pattern=r"^ans:"), group=0)

    # ===================== Other top-level menu =====================
    if _has("menu_router"):
        _add(
            CallbackQueryHandler(
                menu_router,
                pattern=r"^menu_(predict|mock_predict|ask|profile|coach|back)$",
            ),
            group=0,
        )

    # ===================== Admin commands =====================
    if _has("set_round"):          _add(CommandHandler("set_round", set_round), group=5)
    if _has("which_round"):        _add(CommandHandler("which_round", which_round), group=5)
    if _has("list_cutoff_sheets"): _add(CommandHandler("list_sheets", list_cutoff_sheets), group=5)
    if _has("use_cutoff_sheet"):   _add(CommandHandler("use_cutoff_sheet", use_cutoff_sheet), group=5)
    if _has("cutoff_headers"):     _add(CommandHandler("cutoff_headers", cutoff_headers), group=5)
    if _has("set_cutsheet"):       _add(CommandHandler("set_cutsheet", set_cutsheet), group=5)
    if _has("cutdiag"):            _add(CommandHandler("cutdiag", cutdiag), group=5)
    if _has("quota_counts"):       _add(CommandHandler("quota_counts", quota_counts), group=5)
    if _has("cutoff_probe"):       _add(CommandHandler("cutoff_probe", cutoff_probe), group=5)

    # ===================== Ask (Doubt) conversation =====================
    if _has("ask_start", "ask_subject_select", "ask_receive_photo", "ask_receive_text", "cancel") and \
       _has_all("ASK_SUBJECT", "ASK_WAIT"):
        ask_conv = ConversationHandler(
            entry_points=[
                CommandHandler("ask", ask_start),
                CallbackQueryHandler(ask_start, pattern=r"^menu_ask$"),
            ],
            states={
                ASK_SUBJECT: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_subject_select)],
                ASK_WAIT: [
                    MessageHandler(filters.PHOTO, ask_receive_photo),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, ask_receive_text),
                ],
            },
            fallbacks=[CommandHandler("cancel", cancel)],
            name="ask_conv",
            persistent=False,
            per_message=False,
        )
        _add(ask_conv, group=1)

    # ===================== Predictor conversation =====================
    if _has(
        "predict_start", "on_air", "on_quota", "on_category",
        "on_domicile", "on_pg_req_cb", "on_pg_req",
        "on_bond_avoid_cb", "on_bond_avoid", "on_pref", "cancel_predict"
    ) and _has_all("ASK_AIR", "ASK_MOCK_RANK", "ASK_MOCK_SIZE", "ASK_QUOTA", "ASK_CATEGORY", "ASK_DOMICILE"):
        predict_conv = ConversationHandler(
            entry_points=[
                CommandHandler("predict", predict_start),
                CallbackQueryHandler(predict_start, pattern=r"^menu_predict$"),
                CommandHandler("mockpredict", predict_mockrank_start),
                CallbackQueryHandler(predict_mockrank_start, pattern=r"^menu_predict_mock$"),
            ],
            states={
                ASK_AIR:        [MessageHandler(filters.TEXT & ~filters.COMMAND, on_air)],
                ASK_MOCK_RANK:  [MessageHandler(filters.TEXT & ~filters.COMMAND, predict_mockrank_collect_rank)],
                ASK_MOCK_SIZE:  [MessageHandler(filters.TEXT & ~filters.COMMAND, predict_mockrank_collect_size)],
                ASK_QUOTA:      [MessageHandler(filters.TEXT & ~filters.COMMAND, on_quota)],
                ASK_CATEGORY:   [MessageHandler(filters.TEXT & ~filters.COMMAND, on_category)],
                ASK_DOMICILE:   [MessageHandler(filters.TEXT & ~filters.COMMAND, on_domicile)],
            },
            fallbacks=[CommandHandler("cancel", cancel_predict)],
            name="predict_conv",
            persistent=False,
            per_message=False,
        )
        _add(predict_conv, group=3)

    # ===================== Profile conversation =====================
    if _has(
        "setup_profile", "profile_menu", "profile_set_category", "profile_set_domicile",
        "profile_set_pref", "profile_set_email", "profile_set_mobile", "profile_set_primary", "cancel"
    ) and _has_all(
        "PROFILE_MENU", "PROFILE_SET_CATEGORY", "PROFILE_SET_DOMICILE",
        "PROFILE_SET_PREF", "PROFILE_SET_EMAIL", "PROFILE_SET_MOBILE", "PROFILE_SET_PRIMARY"
    ):
        profile_conv = ConversationHandler(
            entry_points=[
                CommandHandler("profile", setup_profile),
                CallbackQueryHandler(setup_profile, pattern=r"^menu_profile$"),
            ],
            states={
                PROFILE_MENU:         [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_menu)],
                PROFILE_SET_CATEGORY: [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_set_category)],
                PROFILE_SET_DOMICILE: [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_set_domicile)],
                PROFILE_SET_PREF:     [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_set_pref)],
                PROFILE_SET_EMAIL:    [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_set_email)],
                PROFILE_SET_MOBILE: [
                    MessageHandler(filters.CONTACT, profile_set_mobile),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, profile_set_mobile),
                ],
                PROFILE_SET_PRIMARY:  [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_set_primary)],
            },
            fallbacks=[CommandHandler("cancel", cancel)],
            name="profile_conv",
            persistent=False,
        )
        _add(profile_conv, group=4)

    # ===================== AI Coach =====================
    if _has("coach_start", "coach_adjust_cb", "coach_save_cb"):
        _add(CommandHandler("coach", coach_start), group=0)
        _add(CallbackQueryHandler(coach_start, pattern=r"^menu_coach$"), group=0)
    if _has("coach_notes_cb"):
        _add(CallbackQueryHandler(coach_notes_cb, pattern=r"^coach_notes:v1$"), group=0)
    if _has("ai_notes_from_shortlist"):
        _add(CallbackQueryHandler(ai_notes_from_shortlist, pattern=r"^ai_notes$"), group=0)

    # ===================== Safety net =====================
    if _has("_unknown_cb"):
        _add(CallbackQueryHandler(_unknown_cb), group=9)

    log.info("✅ Handlers registered")








