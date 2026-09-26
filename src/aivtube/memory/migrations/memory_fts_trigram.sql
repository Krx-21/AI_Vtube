-- The §6 FTS5 trigram index over memory(text, subject). Idempotent; applied at every open when
-- fts5_trigram_available() is true (SQLite >= 3.34). Without trigram the triggers are dropped
-- (they would make every memory write fail) and meta.fts_stale is set, so the index is rebuilt
-- the next time a trigram-capable SQLite opens the file.
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(text, subject, content='memory', content_rowid='id', tokenize='trigram');
CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN INSERT INTO memory_fts(rowid,text,subject) VALUES(new.id,new.text,new.subject); END;
CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN INSERT INTO memory_fts(memory_fts,rowid,text,subject) VALUES('delete',old.id,old.text,old.subject); END;
CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
  INSERT INTO memory_fts(memory_fts,rowid,text,subject) VALUES('delete',old.id,old.text,old.subject);
  INSERT INTO memory_fts(rowid,text,subject) VALUES(new.id,new.text,new.subject); END;
