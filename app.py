import os
from fastapi import FastAPI, Request, HTTPException, Header
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from bot import register_handlers
register_handlers(tg)

# Required env vars (you'll set these on Render):
# TELEGRAM_TOKEN  -> from BotFather
# WEBHOOK_SECRET  -> any long random string; Telegram will echo it as a header

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN env var is required")

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

app = FastAPI()

# Build async PTB app
tg = Application.builder().token(TELEGRAM_TOKEN).concurrent_updates(True).build()
from bot import register_handlers       # NEW
register_handlers(tg) 

# --- Minimal handler (we'll plug Aceit logic later) ---
async def start(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello from Aceit! Bot is alive ✅")

tg.add_handler(CommandHandler("start", start))

# --- Proper lifecycle for PTB inside FastAPI ---
@app.on_event("startup")
async def on_startup():
    await tg.initialize()
    await tg.start()

@app.on_event("shutdown")
async def on_shutdown():
    await tg.stop()
    await tg.shutdown()

# --- Webhook endpoint for Telegram ---
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
