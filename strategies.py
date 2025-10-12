from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Defaults to the same folder as this file unless overridden.
_BASE = Path(__file__).resolve().parent
_DEFAULT_FILE = _BASE / "strategies.json"
_FILE = Path(os.getenv("STRATEGIES_FILE") or _DEFAULT_FILE).expanduser()

_LAST_ERROR: str | None = None
_LOADED = False

_PLANS: Dict[int, dict] = {}
_META: Dict[str, object] = {}

def _resolve_file(path: Optional[str]) -> Path:
    if path:
        return Path(path)
    env_path = os.getenv("STRATEGIES_FILE")
    if env_path:
        return Path(env_path)
    return _FILE

def load_strategies(path: Optional[str] = None) -> None:
    global _FILE, _LOADED

    target = _resolve_file(path).expanduser()
    if target != _FILE:
        _FILE = target
        _PLANS.clear()
        _META.clear()
        _LOADED = False

    if _LOADED:
        return

    if not _FILE.exists():
        # Helpful error so you see it in Render logs
        raise FileNotFoundError(f"strategies.json not found at {_FILE}. "
                                f"Make sure it is committed at repo root.")

    with _FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    _PLANS.clear()
    for plan in data["plans"]:
        _PLANS[int(plan["plan_id"])] = plan
    _META.clear()
    for k, v in data.items():
        if k != "plans":
            _META[k] = v
    _LOADED = True

def get_plan(plan_id: int) -> Optional[dict]:
    if not _PLANS:
        load_strategies()
    return _PLANS.get(int(plan_id))

def get_menu() -> List[Tuple[int, str]]:
    if not _PLANS:
        load_strategies()
    return [(pid, _PLANS[pid]["title"]) for pid in sorted(_PLANS.keys())]
def strategies_path() -> str:
    try:
        return str(_FILE.resolve())
    except Exception:
        return str(_FILE)
        
def reload_strategies(file_path: str | None = None) -> int:
    """Force reload and return count; records last error string if any."""
    global _STRATS, _BY_ID, _LAST_ERROR
    _STRATS, _BY_ID = [], {}
    _LAST_ERROR = None
    try:
        load_strategies(file_path)
        return len(_STRATS)
    except Exception as e:
        _LAST_ERROR = f"{type(e).__name__}: {e}"
        _STRATS, _BY_ID = [], {}
        return 0

def strategies_count() -> int:
    load_strategies()
    return len(_STRATS)

def strategies_last_error() -> str | None:
    return _LAST_ERROR



