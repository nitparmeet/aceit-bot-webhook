import os
import logging
from fastapi import FastAPI, Request, HTTPException, Header
from telegram import Update
from telegram.ext import Application

# ----------------- Config -----------------
# Required: set these in Render → Environment
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]          # your bot token
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")       # optional; leave blank if not using

# Basic logging (Render shows stdout)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("aceit-bot")

# ----------------- FastAPI app -----------------
app = FastAPI()

# Create Telegram Application ONCE (we feed it updates manually)
tg = Application.builder().token(TELEGRAM_TOKEN).concurrent_updates(True).build()

# Import bot hooks (fail fast so errors are visible)
from bot import register_handlers  # must exist in bot.py
register_handlers(tg)
log.info("✅ Handlers registered")

# Optional: dataset/bootstrap loader (if you added it in bot.py)
try:
    from bot import on_startup as bot_on_startup  # async def on_startup(app: Application)
except Exception:
    bot_on_startup = None

# ----------------- Lifecycle -----------------
@app.on_event("startup")
async def _startup():
    # Initialize PTB app (required even in manual-webhook mode)
    await tg.initialize()
    await tg.start()
    log.info("✅ Telegram Application started")

    # Run your dataset/bootstrap logic once (optional)
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

# ----------------- Telegram webhook -----------------
@app.post("/telegram")
async def telegram_webhook(
    req: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    # If you set WEBHOOK_SECRET, require Telegram to send it
    if WEBHOOK_SECRET and (x_telegram_bot_api_secret_token != WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Bad secret header")

    data = await req.json()
    update = Update.de_json(data, tg.bot)
    await tg.process_update(update)
    return {"ok": True}

# ----------------- Health & Home -----------------
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/")
async def home():
    return {"ok": True, "service": "aceit-bot-webhook", "health": "/healthz"}

# Optional: silence HEAD / 405s in logs
@app.head("/")
async def home_head():
    return ""
