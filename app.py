import os
from fastapi import FastAPI, Request, HTTPException, Header
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from bot import register_handlers
register_handlers(tg)


TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

app = FastAPI()

# 1) create tg first
tg = Application.builder().token(TELEGRAM_TOKEN).concurrent_updates(True).build()

# 2) THEN import and register your handlers
from bot import register_handlers      # NEW (after tg is defined)
register_handlers(tg)                  # NEW

@app.on_event("startup")
async def on_startup():
    await tg.initialize()
    await tg.start()

@app.on_event("shutdown")
async def on_shutdown():
    await tg.stop()
    await tg.shutdown()

@app.post("/telegram")
async def telegram_webhook(req: Request, x_telegram_bot_api_secret_token: str | None = Header(default=None)):
    if WEBHOOK_SECRET and (x_telegram_bot_api_secret_token != WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Bad secret header")
    data = await req.json()
    update = Update.de_json(data, tg.bot)
    await tg.process_update(update)
    return {"ok": True}

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
