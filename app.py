import os

import random
import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from db import init_db, close_db, record_usage_event

from fastapi import FastAPI, Request, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from telegram import Update
from telegram.ext import Application

# -------- Config --------
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

# -------- FastAPI app --------
app = FastAPI(title="aceit-bot-webhook")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Telegram Application (single instance) --------
async def _post_init(app: Application):
    """Runs after Application.initialize(). Logs the bot identity."""
    me = await app.bot.get_me()
    log.info("🤖 Bot online as @%s (id=%s)", me.username, me.id)

tg: Application = (
    Application.builder()
    .token(TELEGRAM_TOKEN)
    .concurrent_updates(True)
    .build()
)
tg.post_init = _post_init  # <-- _post_init is defined above (no NameError now)

# Import handler registrars from bot.py and attach them to `tg`
from bot import register_handlers  # must accept (app: Application)
register_handlers(tg)
log.info("✅ Handlers registered")

# Optional startup bootstrap from bot.py
try:
    from bot import on_startup as bot_on_startup  # async def on_startup(app: Application)
except Exception:
    bot_on_startup = None

# -------- quiz pool helpers --------
_POOL: List[Dict[str, Any]] = []
_INDEX: Dict[str, Dict[str, Any]] = {}


def _load_pool() -> None:
    global _POOL, _INDEX
    from quiz_loader import load_quiz_file

    try:
        questions = load_quiz_file(QUIZ_FILE)
    except Exception as exc:
        log.warning(
            "quiz file %s could not be loaded (%s); continuing without quiz questions",
            QUIZ_FILE,
            exc,
        )
        _POOL, _INDEX = [], {}
        return
   

     _POOL = questions

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

# -------- lifecycle --------
@app.on_event("startup")
async def _startup():
    await init_db()
    await tg.initialize()  # triggers tg.post_init
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

# -------- webhook --------
@app.post("/telegram")
async def telegram_webhook(
    req: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
    if WEBHOOK_SECRET and (x_telegram_bot_api_secret_token != WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Bad secret header")
    data = await req.json()
    update = Update.de_json(data, tg.bot)

    user = update.effective_user
    chat = update.effective_chat

    async def _log_usage():
        try:
            user_id = str(user.id) if user and getattr(user, "id", None) is not None else None
            chat_id = str(chat.id) if chat and getattr(chat, "id", None) is not None else None
            if not user_id and not chat_id:
                return
            event_type = _classify_event(update)
            meta = _event_meta(update)
            await record_usage_event(user_id, chat_id, event_type, meta)
        except Exception:
            log.exception("usage logging failed")

    asyncio.create_task(_log_usage())

    await tg.process_update(update)
    return {"ok": True}

def _classify_event(update: Update) -> str:
    if update.message:
        if update.message.text:
            return "message:text"
        if update.message.photo:
            return "message:photo"
        if update.message.document:
            return "message:document"
        return "message"
    if update.callback_query:
        return "callback"
    if update.inline_query:
        return "inline_query"
    if update.chosen_inline_result:
        return "inline_result"
    if update.poll_answer:
        return "poll_answer"
    return "update"

def _event_meta(update: Update) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    msg = update.message
    if msg and msg.text:
        meta["text"] = msg.text[:160]
    cq = update.callback_query
    if cq and cq.data:
        meta["callback_data"] = cq.data[:160]
    return meta

# -------- quiz endpoints --------
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

# -------- health/home --------
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/")
async def home():
    return {"ok": True, "service": "aceit-bot-webhook", "health": "/healthz"}

@app.head("/")
async def home_head():
    return ""
