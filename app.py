# app.py
import os
import json
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from db import init_db, close_db  

from fastapi import FastAPI, Request, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from telegram import Update
from telegram.ext import Application

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

BASE_DIR = Path(__file__).parent
QUIZ_FILE = Path(os.environ.get("QUIZ_FILE", BASE_DIR / "quiz.json"))
RANDOM_SEED = os.environ.get("QUIZ_RANDOM_SEED")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("aceit-bot")

app = FastAPI(title="aceit-bot-webhook")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the Telegram application once
tg: Application = Application.builder().token(TELEGRAM_TOKEN).concurrent_updates(True).build()

# Import handler registrars from bot.py and attach them to `tg`
from bot import register_handlers  # must accept (app: Application)
register_handlers(tg)
log.info("✅ Handlers registered")

# Optional startup bootstrap from bot.py
try:
    from bot import on_startup as bot_on_startup  # async def on_startup(app: Application)
except Exception:
    bot_on_startup = None

# -------------- quiz pool helpers --------------
_POOL: List[Dict[str, Any]] = []
_INDEX: Dict[str, Dict[str, Any]] = {}

def _validate_question(q: Dict[str, Any], i: int) -> Optional[str]:
    if "id" not in q or not q["id"]:
        return f"[{i}] missing id"
    if "options" not in q or not isinstance(q["options"], list) or len(q["options"]) < 2:
        return f"[{q.get('id')}] options must be a list with >= 2 items"
    ai = q.get("answer_index")
    if not isinstance(ai, int) or ai < 0 or ai >= len(q["options"]):
        return f"[{q.get('id')}] invalid answer_index"
    if q.get("difficulty") not in (1, 2, 3):
        return f"[{q.get('id')}] difficulty must be 1|2|3"
    if "subject" not in q or not q["subject"]:
        return f"[{q.get('id')}] subject required"
    return None

def _load_pool() -> None:
    global _POOL, _INDEX
    if not QUIZ_FILE.exists():
        raise FileNotFoundError(f"quiz file not found at: {QUIZ_FILE}")
    data = json.load(QUIZ_FILE.open("r", encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("quiz.json must be a flat JSON array")

    seen, errors = set(), []
    for i, q in enumerate(data):
        err = _validate_question(q, i)
        if err:
            errors.append(err); continue
        if q["id"] in seen:
            errors.append(f"duplicate id: {q['id']}"); continue
        seen.add(q["id"])

    if errors:
        raise ValueError("quiz.json validation errors:\n- " + "\n- ".join(errors))

    _POOL = data
    _INDEX = {q["id"]: q for q in _POOL}
    log.info("✅ Loaded %d quiz questions from %s", len(_POOL), QUIZ_FILE)

def _ensure_loaded():
    if not _POOL:
        if RANDOM_SEED:
            random.seed(RANDOM_SEED)
        _load_pool()

def _pick_questions(
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
        out = list(out); random.shuffle(out)
    return out[:count]

def _strip_answers(qs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "id": q["id"],
        "subject": q.get("subject"),
        "topic": q.get("topic"),
        "difficulty": q.get("difficulty"),
        "question": q.get("question"),
        "options": q.get("options"),
        "tags": q.get("tags", []),
    } for q in qs]

# -------------- lifecycle --------------
@app.on_event("startup")
async def _startup():
    await init_db()  
    await tg.initialize()
    await tg.start()
    log.info("✅ Telegram Application started")

    try:
        _ensure_loaded()
    except Exception:
        log.exception("❌ Failed loading quiz.json")

    if bot_on_startup is not None:
        try:
            await bot_on_startup(tg)
            log.info("✅ Startup tasks complete")
        except Exception:
            log.exception("❌ bot_on_startup failed")

@app.on_event("shutdown")
async def _shutdown():
    await tg.stop()
    await tg.shutdown()
    log.info("🛑 Telegram Application stopped")

    await close_db()   

# -------------- webhook --------------
@app.post("/telegram")
async def telegram_webhook(
    req: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
    if WEBHOOK_SECRET and (x_telegram_bot_api_secret_token != WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Bad secret header")
    data = await req.json()
    update = Update.de_json(data, tg.bot)
    await tg.process_update(update)
    return {"ok": True}

# -------------- quiz endpoints --------------
@app.get("/quiz")
async def get_quiz(
    count: int = Query(5, pattern="^(5|10)$"),
    subject: Optional[str] = Query(default=None),
    difficulty: Optional[int] = Query(default=None, ge=1, le=3),
    tags: Optional[str] = Query(default=None, description="comma separated"),
    reveal: Optional[int] = Query(default=0, ge=0, le=1),
):
    try:
        _ensure_loaded()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    tags_any = [t.strip() for t in tags.split(",")] if tags else None
    qs = _pick_questions(_POOL, subject=subject, difficulty=difficulty, tags_any=tags_any, count=count, shuffle=True)
    return JSONResponse(content=qs if reveal == 1 else _strip_answers(qs))

@app.post("/grade")
async def grade_quiz(payload: Dict[str, Any]):
    try:
        _ensure_loaded()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    if not payload or "responses" not in payload or not isinstance(payload["responses"], list):
        raise HTTPException(status_code=400, detail="Body must include 'responses' array")

    score, details = 0, []
    for item in payload["responses"]:
        qid = item.get("id")
        user_ai = item.get("answer_index")
        q = _INDEX.get(qid)
        if not q:
            details.append({"id": qid, "correct": False, "error": "unknown question id"})
            continue
        correct_ai = q["answer_index"]
        is_correct = (user_ai == correct_ai)
        if is_correct:
            score += 1
        details.append({
            "id": qid,
            "correct": is_correct,
            "correct_index": correct_ai,
            "explanation": q.get("explanation"),
            "subject": q.get("subject"),
            "topic": q.get("topic"),
        })

    return JSONResponse(content={"score": score, "total": len(payload["responses"]), "details": details})

# -------------- health/home --------------
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/")
async def home():
    return {"ok": True, "service": "aceit-bot-webhook", "health": "/healthz"}

@app.head("/")
async def home_head():
    return ""
