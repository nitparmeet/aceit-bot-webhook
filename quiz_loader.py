"""
Helpers to read and normalise quiz question datasets.

The production quiz file has drifted across formats (flat arrays, subject-keyed
dicts, legacy answer fields, merge-conflict markers).  We sanitise the input so
callers always receive a flat list of question dicts with the canonical fields:
  id, subject, topic, difficulty (int 1-3), question, options, answer_index,
  tags, explanation.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

log = logging.getLogger("aceit-bot")

DIFF_MAP = {
    "low": 1,
    "easy": 1,
    "1": 1,
    "mid": 2,
    "medium": 2,
    "moderate": 2,
    "2": 2,
    "high": 3,
    "hard": 3,
    "3": 3,
}


def load_quiz_file(path: str | Path) -> List[Dict[str, Any]]:
    """
    Read `quiz.json`, handle legacy structures, and return a flat list
    of validated question dicts. Raises ValueError on unrecoverable issues.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"quiz file not found at: {p}")

    raw = p.read_text(encoding="utf-8")
    cleaned = raw.strip()
    if not cleaned:
        log.warning("quiz file %s is empty; continuing with no quiz questions", p)
        return []

    cleaned = _strip_conflict_markers(cleaned)

    payload = _loads_loose(cleaned)
    if payload is None:
        raise ValueError(f"quiz file {p} is not valid JSON even after cleanup")

    questions, errors = _normalise_payload(payload)
    if errors:
        if questions:
            preview = "\n- ".join(errors[:5])
            if len(errors) > 5:
                preview += f"\n- ... (+{len(errors) - 5} more issues)"
            log.warning("quiz.json contained malformed entries:\n- %s", preview)
        else:
            raise ValueError("quiz.json validation errors:\n- " + "\n- ".join(errors))
    return questions


def _strip_conflict_markers(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith(("<<<<<<<", "=======", ">>>>>>>")):
            continue
        lines.append(line)
    return "\n".join(lines)


def _loads_loose(text: str) -> Any | None:
    """
    Try to parse with json.loads; on failure attempt to decode the first JSON
    value (useful when extra data follows the primary document).
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        try:
            data, _ = json.JSONDecoder().raw_decode(text)
            return data
        except Exception:
            log.warning("quiz JSON parse failed: %s", exc)
            return None


def _normalise_payload(payload: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    if isinstance(payload, list):
        return _normalise_list(payload)
    if isinstance(payload, dict):
        for key in WRAPPER_KEYS:
            nested = payload.get(key)
            if nested is not None:
                qs, errs = _normalise_payload(nested)
                if qs or errs:
                    return qs, errs  
        return _normalise_subject_map(payload)
    return [], ["root must be a list or an object mapping subjects -> questions"]


def _normalise_list(items: Iterable[Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    out: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(items, start=1):
        q, err = _coerce_question(item, prefix=None, index=idx)
        if err:
            errors.append(f"[{idx}] {err}")
            continue
        if q:
            if q["id"] in seen_ids:
                errors.append(f"[{idx}] duplicate id: {q['id']}")
                continue
            seen_ids.add(q["id"])
            out.append(q)
    return out, errors


def _normalise_subject_map(data: Dict[Any, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    out: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen_ids: set[str] = set()

    for subject, arr in data.items():
        key_norm = str(subject or "").strip()
        lower_key = key_norm.lower()
        if lower_key in WRAPPER_KEYS:
            qs, errs = _normalise_payload(arr)
            for q in qs:
                if q["id"] in seen_ids:
                    errors.append(f"[{q['id']}] duplicate id")
                    continue
        if not isinstance(arr, list):
            extracted = _extract_question_block(arr)
            if extracted is None:
                errors.append(f"[{subject}] expected list of questions")
                continue
            arr = extracted
                seen_ids.add(q["id"])
                out.append(q)
            errors.extend(errs)
            continue
        prefix = str(subject or "GEN").strip().upper()[:3] or "GEN"
        for idx, item in enumerate(arr, start=1):
            q, err = _coerce_question(item, prefix=prefix, index=idx, subject_hint=subject)
            if err:
                errors.append(f"[{subject} #{idx}] {err}")
                continue
            if q:
                if q["id"] in seen_ids:
                    errors.append(f"[{q['id']}] duplicate id")
                    continue
                seen_ids.add(q["id"])
                out.append(q)
    return out, errors


def _coerce_question(
    item: Any,
    *,
    prefix: str | None,
    index: int,
    subject_hint: Any = None,
) -> Tuple[Dict[str, Any] | None, str | None]:
    if not isinstance(item, dict):
        return None, "question must be an object"

    subject = _coerce_str(item.get("subject")) or _coerce_str(subject_hint) or "General"
    qid = _coerce_str(item.get("id"))
    if not qid:
        base = prefix or subject.upper()[:3] or "GEN"
        qid = f"{base}-{index:04d}"

    question = _coerce_str(item.get("question"))
    if not question:
        return None, "missing question text"

    options_raw = item.get("options")
    options = _coerce_options(options_raw)
    if len(options) < 2:
        return None, "options must be a list with >= 2 choices"
    options = [_coerce_str(opt) for opt in options]

    ans_index = _coerce_answer_index(item, options)
    if ans_index is None:
        return None, "could not determine correct answer index"
    if not (0 <= ans_index < len(options)):
        return None, f"answer_index {ans_index} out of range for {len(options)} options"

    difficulty = _coerce_difficulty(item.get("difficulty"))

    tags = _coerce_tags(item.get("tags"))

    explanation = _coerce_str(item.get("explanation"))

    return (
        {
            "id": qid,
            "subject": subject,
            "topic": _coerce_str(item.get("topic")),
            "difficulty": difficulty,
            "question": question,
            "options": options,
            "answer_index": ans_index,
            "tags": tags,
            "explanation": explanation,
        },
        None,
    )


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        s = str(value).strip()
    except Exception:
        return ""
    return s


def _coerce_answer_index(item: Dict[str, Any], options: List[str]) -> int | None:
    ans = item.get("answer_index")
    if isinstance(ans, bool):
        ans = int(ans)
    if isinstance(ans, int):
        return ans
    if isinstance(ans, float) and math.isfinite(ans):
        return int(ans)

    if isinstance(ans, str) and ans.strip().isdigit():
        return int(ans.strip())

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ans_label = item.get("answer") or item.get("correct") or item.get("correct_option")
    if isinstance(ans_label, (int, float)) and not isinstance(ans_label, bool):
        idx = int(ans_label)
        # assume 1-based
        if idx >= 1:
            return idx - 1
        return idx
    if isinstance(ans_label, str):
        raw = ans_label.strip()
        if raw.isdigit():
            idx = int(raw)
            return idx - 1 if idx >= 1 else idx
        if len(raw) == 1 and raw.upper() in letters:
            idx = letters.index(raw.upper())
            if idx < len(options):
                return idx  
        # fall back to matching option text (case-insensitive)
        for i, opt in enumerate(options):
            if opt.lower() == raw.lower():
                return i

    return None


def _coerce_difficulty(value: Any) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        iv = int(value)
        if iv in (1, 2, 3):
            return iv
    if isinstance(value, str):
        key = value.strip().lower()
        if key in DIFF_MAP:
            return DIFF_MAP[key]
    return 2  # default to moderate


def _coerce_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        segs = [seg.strip() for seg in value.split(",")]
        return [seg for seg in segs if seg]
    return []

def _extract_question_block(node: Any) -> List[Any] | None:
    if node is None:
        return None
    if isinstance(node, list):
        return node
    if isinstance(node, dict):
        if "question" in node and ("options" in node or "answer" in node or "answer_index" in node):
            return [node]
        for key in WRAPPER_KEYS:
            child = node.get(key)
            res = _extract_question_block(child)
            if res:
                return res
    return None
def _coerce_options(raw: Any) -> List[str]:
    if isinstance(raw, list):
        cleaned = []
        for item in raw:
            if isinstance(item, dict) and "text" in item:
                cleaned.append(_coerce_str(item.get("text")))
            else:
                cleaned.append(_coerce_str(item))
        return [opt for opt in cleaned if opt]
    if isinstance(raw, dict):
        ordered_keys = sorted(raw.keys())
        cleaned = []
        for key in ordered_keys:
            val = raw[key]
            if isinstance(val, dict) and "text" in val:
                cleaned.append(_coerce_str(val.get("text")))
            else:
                cleaned.append(_coerce_str(val))
        return [opt for opt in cleaned if opt]
    if isinstance(raw, str):
        parts = [seg.strip() for seg in raw.split("\n") if seg.strip()]
        if len(parts) >= 2:
            return parts
    return []
WRAPPER_KEYS = {
    "subjects",
    "subject_map",
    "subjectWise",
    "subjectwise",
    "sections",
    "section_map",
    "data",
    "payload",
    "questions",
    "quiz",
    "items",
    "records",
    "results",
    "entries",
    "list",
}

