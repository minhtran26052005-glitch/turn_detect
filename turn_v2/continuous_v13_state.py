"""Atomic state, immutable evaluation inputs and accumulated crawl data."""
from contextlib import contextmanager
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3
import time


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(path)


class Store:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,timeout=30);self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS samples(key TEXT PRIMARY KEY,language TEXT,source_id TEXT,split TEXT,label TEXT,record TEXT,cycle INTEGER);
        CREATE TABLE IF NOT EXISTS sources(source_id TEXT PRIMARY KEY,language TEXT,record TEXT,cycle INTEGER);
        CREATE TABLE IF NOT EXISTS protected(key TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS protected_sources(source_id TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,at REAL,event TEXT,record TEXT);
        ''');self.db.commit()

    def get(self,key,default=None):
        row=self.db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone();return json.loads(row[0]) if row else default

    def put(self,key,value):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,json.dumps(value,ensure_ascii=False)));self.db.commit()

    def protect(self,keys,sources=()):
        self.db.executemany('INSERT OR IGNORE INTO protected VALUES (?)',((k,) for k in keys))
        self.db.executemany('INSERT OR IGNORE INTO protected_sources VALUES (?)',((s,) for s in sources));self.db.commit()

    def allowed(self,key,source_id,split='train'):
        if self.db.execute('SELECT 1 FROM samples WHERE key=?',(key,)).fetchone():return False
        if split=='train' and (self.db.execute('SELECT 1 FROM protected WHERE key=?',(key,)).fetchone() or self.db.execute('SELECT 1 FROM protected_sources WHERE source_id=?',(source_id,)).fetchone()):return False
        return True

    def add(self,key,row,split,cycle):
        if not self.allowed(key,row['source_id'],split):return False
        self.db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?)',(key,row['language'],row['source_id'],split,row['judge_label'],json.dumps(row,ensure_ascii=False),cycle))
        if split=='semantic_dev':
            self.db.execute('INSERT OR IGNORE INTO protected VALUES (?)',(key,));self.db.execute('INSERT OR IGNORE INTO protected_sources VALUES (?)',(row['source_id'],))
        self.db.commit();return True

    def rows(self,language,split,limit=60000):
        return [json.loads(r[0]) for r in self.db.execute('SELECT record FROM samples WHERE language=? AND split=? ORDER BY cycle DESC,key LIMIT ?',(language,split,limit))]

    def counts(self):return [dict(zip(['language','split','label','samples'],row)) for row in self.db.execute('SELECT language,split,label,COUNT(*) FROM samples GROUP BY language,split,label')]

    def source(self,row,cycle):
        changed=self.db.execute('INSERT OR IGNORE INTO sources VALUES (?,?,?,?)',(row['source_id'],row['language'],json.dumps(row,ensure_ascii=False),cycle)).rowcount;self.db.commit();return bool(changed)

    def source_seen(self,source_id):return bool(self.db.execute('SELECT 1 FROM sources WHERE source_id=?',(source_id,)).fetchone())

    def event(self,event,value):
        self.db.execute('INSERT INTO events(at,event,record) VALUES (?,?,?)',(time.time(),event,json.dumps(value,ensure_ascii=False)));self.db.commit()

    def backup(self,path):
        target=sqlite3.connect(path)
        with target:self.db.backup(target)
        target.close()


@contextmanager
def singleton(path):
    """An OS lock survives stale PID files and is released on process death."""
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Continuous trainer already owns this run; no duplicate launched')
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)
