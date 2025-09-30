# db.py
import os, asyncpg
from typing import Optional

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
