-- users / profiles
create table if not exists users (
  user_id        text primary key,
  username       text,
  first_name     text,
  last_name      text,
  city           text,
  state          text,
  class_grade    text,
  target_exam    text,
  created_at     timestamptz not null default now(),
  updated_at     timestamptz not null default now()
);

-- questions (if you already have IDs elsewhere, you can leave this empty & just store the id strings in answers)
create table if not exists questions (
  question_id    text primary key,
  subject        text not null,
  topic          text,
  difficulty_tag text,
  correct_option text not null
);

-- quiz/test sessions
create table if not exists quiz_sessions (
  session_id     uuid primary key default gen_random_uuid(),
  user_id        text not null references users(user_id),
  mode           text not null,
  subject        text,
  started_at     timestamptz not null default now(),
  finished_at    timestamptz
);

-- answers (1 row per question answered)
create table if not exists answers (
  session_id     uuid references quiz_sessions(session_id) on delete cascade,
  question_id    text,
  chosen_option  text,
  is_correct     boolean,
  time_ms        integer,
  answered_at    timestamptz not null default now(),
  primary key (session_id, question_id)
);

-- results (aggregated per session)
create table if not exists results (
  session_id     uuid primary key references quiz_sessions(session_id) on delete cascade,
  user_id        text not null references users(user_id),
  total_q        integer not null,
  correct_q      integer not null,
  score_pct      numeric(5,2) generated always as (100.0*correct_q/NULLIF(total_q,0)) stored,
  finished_at    timestamptz not null
);

-- helpful indexes
create index if not exists answers_question_idx on answers (question_id, is_correct);
create index if not exists answers_session_idx  on answers (session_id);
create index if not exists results_user_idx     on results (user_id, finished_at desc);
create index if not exists sessions_user_idx    on quiz_sessions (user_id, started_at desc);

-- usage analytics
create table if not exists bot_usage_events (
  event_id    uuid primary key default gen_random_uuid(),
  user_id     text,
  chat_id     text,
  event_type  text,
  meta        jsonb,
  created_at  timestamptz not null default now()
);

create index if not exists bot_usage_events_created_idx on bot_usage_events (created_at);
create index if not exists bot_usage_events_user_idx    on bot_usage_events (user_id, created_at desc);
