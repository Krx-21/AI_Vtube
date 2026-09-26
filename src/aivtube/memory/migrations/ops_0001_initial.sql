-- data/ops.db, version 1: the ARCHITECTURE.md §6 schema.
CREATE TABLE turn_trace(turn_id TEXT PRIMARY KEY, character TEXT, session_id INTEGER, kind TEXT, provider TEXT, tts_backend TEXT,
  tts_identity TEXT, stages TEXT NOT NULL, ttfa_ms REAL, tokens_out INTEGER, tok_s REAL, prompt_n INTEGER, cache_n INTEGER,
  speculative INTEGER, opener TEXT, outcome TEXT);
CREATE TABLE moderation_log(id INTEGER PRIMARY KEY, ts REAL, character TEXT, direction TEXT, source TEXT, tier TEXT,
  category TEXT, rule TEXT, verdict TEXT, text_masked TEXT, text_sha256 TEXT, author TEXT, turn_id TEXT);
CREATE TABLE tool_audit(id INTEGER PRIMARY KEY, ts REAL, character TEXT, turn_id TEXT, tool TEXT, args TEXT, verdict TEXT,
  result TEXT, approved_by TEXT, dry_run INTEGER);
CREATE TABLE op_audit(id INTEGER PRIMARY KEY, ts REAL, operator TEXT, command TEXT, args TEXT, result TEXT, latency_ms REAL);
CREATE TABLE job(id INTEGER PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL, payload TEXT,
  attempts INTEGER NOT NULL DEFAULT 0, next_run_at REAL, last_error TEXT);
