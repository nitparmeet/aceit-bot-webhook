# db.py
import os, json, asyncpg
from typing import Optional, Any, Dict
from asyncpg import exceptions as pg_exc
_POOL: Optional[asyncpg.Pool] = None

async def init_db():
    global _POOL
    if _POOL is None:
        _POOL = await asyncpg.create_pool(
            os.environ["DATABASE_URL"],
            min_size=1, max_size=5,
            command_timeout=10
        )

async def close_db():
    global _POOL
    if _POOL:
        await _POOL.close()
        _POOL = None

def _pool() -> asyncpg.Pool:
    if _POOL is None:
        raise RuntimeError("DB pool not initialized")
    return _POOL

# ---- user/profile ----
async def upsert_user(u, **extra):
    q = """
    insert into users (user_id, username, first_name, last_name, city, state, class_grade, target_exam)
    values ($1,$2,$3,$4,$5,$6,$7,$8)
    on conflict (user_id) do update set
      username=excluded.username,
      first_name=excluded.first_name,
      last_name=excluded.last_name,
      city=coalesce(excluded.city, users.city),
      state=coalesce(excluded.state, users.state),
      class_grade=coalesce(excluded.class_grade, users.class_grade),
      target_exam=coalesce(excluded.target_exam, users.target_exam),
      updated_at=now();
    """
    async with _pool().acquire() as c:
        await c.execute(
            q, str(u.id), u.username, u.first_name, u.last_name,
            extra.get("city"), extra.get("state"),
            extra.get("class_grade"), extra.get("target_exam")
        )

# ---- sessions & answers ----
async def create_session(user_id: str, mode: str, subject: str | None) -> str:
    q = "insert into quiz_sessions (user_id, mode, subject) values ($1,$2,$3) returning session_id"
    async with _pool().acquire() as c:
        row = await c.fetchrow(q, user_id, mode, subject)
        return str(row["session_id"])

async def save_answer(session_id: str, question_id: str, chosen: str, is_correct: bool, time_ms: int | None):
    q = """
    insert into answers (session_id, question_id, chosen_option, is_correct, time_ms)
    values ($1,$2,$3,$4,$5)
    on conflict (session_id, question_id) do update set
      chosen_option=excluded.chosen_option,
      is_correct=excluded.is_correct,
      time_ms=excluded.time_ms,
      answered_at=now();
    """
    async with _pool().acquire() as c:
        await c.execute(q, session_id, question_id, chosen, is_correct, time_ms)

async def finalize_session(session_id: str, user_id: str):
    q = """
    with agg as (
      select count(*) total_q,
             sum(case when is_correct then 1 else 0 end) correct_q
      from answers where session_id=$1
    )
    insert into results (session_id, user_id, total_q, correct_q, finished_at)
    select $1, $2, total_q, correct_q, now() from agg
    on conflict (session_id) do update set
      total_q=excluded.total_q,
      correct_q=excluded.correct_q,
      finished_at=excluded.finished_at;

    update quiz_sessions set finished_at=now() where session_id=$1;
    """
    async with _pool().acquire() as c:
        await c.execute(q, session_id, user_id)

# ---- usage analytics ----
async def record_usage_event(user_id: Optional[str], chat_id: Optional[str], event_type: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """Persist a lightweight usage event for analytics counters."""
    if user_id is None and chat_id is None:
        return

    q = """
    insert into bot_usage_events (user_id, chat_id, event_type, meta)
    values ($1, $2, $3, $4::jsonb)
    """
    payload = json.dumps(meta) if meta else None
    async with _pool().acquire() as c:
        try:
            await c.execute(q, user_id, chat_id, event_type, payload)
        except pg_exc.UndefinedTableError:
            # schema not yet applied; skip quietly
            return


async def fetch_usage_metrics(days: int = 14) -> Dict[str, Any]:
    """Return summary counters plus daily stats for the requested window."""
    days = max(1, min(int(days or 1), 90))
    async with _pool().acquire() as c:
        total_users = await c.fetchval("select count(*) from users")
        try:
            total_requests = await c.fetchval("select count(*) from bot_usage_events")
            active_today = await c.fetchval(
                "select count(distinct user_id) from bot_usage_events where created_at::date = current_date"
            )
            rows = await c.fetch(
                """
                select
                  created_at::date as day,
                  count(distinct user_id) as dau,
                  count(*) as requests
                from bot_usage_events
                where created_at::date >= (current_date - ($1::int - 1))
                group by day
                order by day
                """,
                days,
            )
        except pg_exc.UndefinedTableError:
            return {
                "total_users": int(total_users or 0),
                "total_requests": 0,
                "active_today": 0,
                "daily": [],
            }

    daily = [
        {
            "day": r["day"],
            "dau": int(r["dau"] or 0),
            "requests": int(r["requests"] or 0),
        }
        for r in rows
    ]

    return {
        "total_users": int(total_users or 0),
        "total_requests": int(total_requests or 0),
        "active_today": int(active_today or 0),
        "daily": daily,
    }

  
