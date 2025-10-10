from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Always resolves to the same folder as this file.
_JSON_PATH = Path(__file__).with_name("strategies.json")

_PLANS: Dict[int, dict] = {}
_META: Dict[str, object] = {}

def load_strategies(path: Optional[str] = None) -> None:
    p = Path(path) if path else _JSON_PATH
    if not p.exists():
        # Helpful error so you see it in Render logs
        raise FileNotFoundError(f"strategies.json not found at {p}. "
                                f"Make sure it is committed at repo root.")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    _PLANS.clear()
    for plan in data["plans"]:
        _PLANS[int(plan["plan_id"])] = plan
    _META.clear()
    for k, v in data.items():
        if k != "plans":
            _META[k] = v

def get_plan(plan_id: int) -> Optional[dict]:
    if not _PLANS:
        load_strategies()
    return _PLANS.get(int(plan_id))

def get_menu() -> List[Tuple[int, str]]:
    if not _PLANS:
        load_strategies()
    return [(pid, _PLANS[pid]["title"]) for pid in sorted(_PLANS.keys())]
