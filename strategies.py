from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


_BASE = Path(__file__).parent
_FILE = Path(os.getenv("STRATEGIES_FILE", str(_BASE / "strategies.json")))

_STRATS: List[Dict] = []
_BY_ID: Dict[str, Dict] = {}
_LAST_ERROR: Optional[str] = None
_LAST_RAW_COUNT: int = 0
_LAST_FILTERED_OUT: int = 0


def _as_list(payload) -> List[Dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("strategies", "plans", "macros", "items"):
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
        return [payload]
    return []


def _normalize(item: Dict) -> Optional[Dict]:
    sid = str(item.get("id") or "").strip()
    title = str(item.get("title") or "").strip()
    if not sid or not title:
        return None

    desc = (
        item.get("description")
        or item.get("short_desc")
        or item.get("short_description")
        or item.get("desc")
        or ""
    )
    bullets = item.get("bullets") or item.get("points") or []
    if isinstance(bullets, str):
        bullets = [bullets]

    stories_field = item.get("stories") or []
    stories: List[Dict[str, Any]] = []
    if isinstance(stories_field, list):
        for entry in stories_field:
            if not isinstance(entry, dict):
                continue
            ref = str(entry.get("ref") or entry.get("id") or "").strip()
            if not ref:
                continue
            story_entry: Dict[str, Any] = {"ref": ref}
            use_when = entry.get("use_when")
            if isinstance(use_when, list):
                values = [str(x).strip() for x in use_when if str(x).strip()]
                if values:
                    story_entry["use_when"] = values
            title_val = entry.get("title")
            if isinstance(title_val, str) and title_val.strip():
                story_entry["title"] = title_val.strip()
            stories.append(story_entry)

    norm = {
        "id": sid,
        "title": title,
        "description": str(desc).strip(),
        "bullets": [str(b).strip() for b in bullets if str(b).strip()],
    }
    return norm


def load_strategies(file_path: Optional[str] = None) -> None:
    global _FILE, _STRATS, _BY_ID, _LAST_ERROR, _LAST_RAW_COUNT, _LAST_FILTERED_OUT

    if _STRATS:
        return

    _LAST_ERROR = None
    _LAST_RAW_COUNT = 0
    _LAST_FILTERED_OUT = 0

    if file_path:
        _FILE = Path(file_path)

    if not _FILE.exists():
        _LAST_ERROR = f"File not found: {_FILE}"
        _STRATS, _BY_ID = [], {}
        return

    try:
        text = _FILE.read_text(encoding="utf-8")
        raw = json.loads(text)
        rows = _as_list(raw)
        _LAST_RAW_COUNT = len(rows)

        normed: List[Dict] = []
        for it in rows:
            norm = _normalize(it)
            if norm:
                normed.append(norm)

        _LAST_FILTERED_OUT = _LAST_RAW_COUNT - len(normed)
        _STRATS = normed
        _BY_ID = {d["id"]: d for d in _STRATS}
    except Exception as exc:
        _LAST_ERROR = f"{type(exc).__name__}: {exc}"
        _STRATS, _BY_ID = [], {}


def reload_strategies(file_path: Optional[str] = None) -> int:
    global _STRATS, _BY_ID

    _STRATS, _BY_ID = [], {}
    load_strategies(file_path)
    return len(_STRATS)


def all_strategies() -> List[Dict]:
    load_strategies()
    return list(_STRATS)


def get_strategy(sid: str) -> Optional[Dict]:
    load_strategies()
    return _BY_ID.get(str(sid).strip())


def strategies_path() -> str:
    try:
        return str(_FILE.resolve())
    except Exception:
        return str(_FILE)


def strategies_count() -> int:
    load_strategies()
    return len(_STRATS)


def strategies_last_error() -> Optional[str]:
    return _LAST_ERROR


def strategies_debug_stats() -> Dict[str, int | str]:
    return {
        "path": strategies_path(),
        "raw": _LAST_RAW_COUNT,
        "loaded": len(_STRATS),
        "filtered": _LAST_FILTERED_OUT,
        "error": _LAST_ERROR or "",
    }
