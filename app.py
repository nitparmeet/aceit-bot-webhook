import os
from fastapi import FastAPI, Request, HTTPException, Header
from telegram import Update
from telegram.ext import Application, ContextTypes


TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

app = FastAPI()

# 1) Create the Telegram application FIRST
tg = Application.builder().token(TELEGRAM_TOKEN).concurrent_updates(True).build()

# 2) THEN import and register your handlers
try:
    from bot import register_handlers   # expects a function register_handlers(app)
    register_handlers(tg)
except Exception as e:
    # If bot.py isn't ready yet, keep the app running
    print("Handler registration skipped:", repr(e))

import logging
logger = logging.getLogger("aceit-bot")
logger.info("✅ Handlers registered")

# FastAPI lifecycle hooks
@app.on_event("startup")
async def on_startup():
    await tg.initialize()
    await tg.start()

@app.on_event("shutdown")
async def on_shutdown():
    await tg.stop()
    await tg.shutdown()

# Telegram webhook endpoint
@app.post("/telegram")
async def telegram_webhook(
    req: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    if WEBHOOK_SECRET and (x_telegram_bot_api_secret_token != WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Bad secret header")
    data = await req.json()
    update = Update.de_json(data, tg.bot)
    await tg.process_update(update)
    return {"ok": True}

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# Optional: avoid 404 at /
@app.get("/")
async def home():
    return {"ok": True, "service": "aceit-bot-webhook", "health": "/healthz"}
