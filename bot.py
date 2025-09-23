import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello from Aceit! Bot is alive ✅")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Help: try /menu or send a message.")

def _heavy_work(text: str) -> str:
    # put your real logic here (OpenAI, DB, etc.)
    return f"Echo: {text}"

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    reply = await asyncio.to_thread(_heavy_work, text)  # offload heavy stuff
    await update.message.reply_text(reply)

# IMPORTANT: use the function argument name (e.g., app), not a global 'tg'
def register_handlers(app: Application):
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
