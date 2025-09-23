# bot.py
import os, asyncio
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

# ----- Your feature handlers (examples) -----
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I can do X, Y, Z. Try /menu")

async def menu_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Menu:\n1) Feature A\n2) Feature B\nType your query…")

# Example: heavy/slow work offloaded to a thread (so webhook never hangs)
def _run_heavy_logic(user_text: str) -> str:
    # TODO: put your existing logic here (OpenAI calls, scoring, etc.)
    # Example (if you use OpenAI):
    # from openai import OpenAI
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # r = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role":"user","content":user_text}],
    #     timeout=30
    # )
    # return r.choices[0].message.content
    return f"Echo: {user_text}"

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    # Offload CPU/network work so the webhook stays fast:
    reply = await asyncio.to_thread(_run_heavy_logic, text)
    await update.message.reply_text(reply)

# ----- This is the only function app.py needs -----
def register_handlers(app: Application):
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("menu", menu_cmd))
    # Add all your feature commands here...
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
