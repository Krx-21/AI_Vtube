-- memory/<char>.sqlite, version 1: the ARCHITECTURE.md §6 schema.
-- The FTS5 trigram index (memory_fts and its triggers) lives in memory_fts_trigram.sql and is
-- applied at open time only when this SQLite build has the trigram tokenizer.
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE session(id INTEGER PRIMARY KEY, started_at REAL NOT NULL, ended_at REAL, title TEXT, platforms TEXT, summary TEXT);
CREATE TABLE epoch(id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES session(id), started_at REAL NOT NULL,
  rolling_summary TEXT NOT NULL DEFAULT '', upto_turn_id INTEGER, prefix_hash TEXT, slot_file TEXT);
CREATE TABLE turn(id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES session(id), epoch_id INTEGER NOT NULL REFERENCES epoch(id),
  ts REAL NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','assistant','tool','note')), source TEXT NOT NULL, speaker TEXT,
  text TEXT NOT NULL, heard_text TEXT, interrupted INTEGER NOT NULL DEFAULT 0, filtered INTEGER NOT NULL DEFAULT 0,
  provider TEXT, turn_ref TEXT, tool_calls TEXT, provider_extra TEXT, tokens INTEGER);
CREATE INDEX turn_epoch ON turn(epoch_id, id);
CREATE TABLE memory(id INTEGER PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('core','fact','viewer','episode')),
  slot INTEGER, subject TEXT, platform TEXT, user_id TEXT, text TEXT NOT NULL,
  importance INTEGER NOT NULL DEFAULT 3 CHECK(importance BETWEEN 1 AND 5),
  source TEXT NOT NULL CHECK(source IN ('model','operator','consolidation','import')), origin TEXT NOT NULL DEFAULT '',
  origin_turn_id INTEGER REFERENCES turn(id),
  status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','quarantined','deleted')),
  pinned INTEGER NOT NULL DEFAULT 0, locked INTEGER NOT NULL DEFAULT 0, epoch_seen INTEGER,
  created_at REAL NOT NULL, updated_at REAL NOT NULL, last_used_at REAL, uses INTEGER NOT NULL DEFAULT 0);
CREATE UNIQUE INDEX memory_core_slot ON memory(slot) WHERE kind='core' AND status='active';
CREATE INDEX memory_viewer ON memory(platform, user_id) WHERE kind='viewer' AND status='active';
CREATE TABLE viewer(platform TEXT NOT NULL, user_id TEXT NOT NULL, name TEXT, first_seen REAL, last_seen REAL,
  messages INTEGER NOT NULL DEFAULT 0, picked INTEGER NOT NULL DEFAULT 0, opt_out INTEGER NOT NULL DEFAULT 0,
  muted_until REAL, PRIMARY KEY(platform, user_id));
CREATE TABLE runtime_state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
