"""Server-side ~/.annot_cache/<user>.jsonl write-ahead queue and replay."""
import json
import os
from pathlib import Path
from filelock import FileLock
from .config import USERS
from .db import validate


class Cache:
    def __init__(self, directory, username):
        if username not in USERS:
            raise ValueError('Unknown annotator')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f'{username}.jsonl'
        self.lock = FileLock(str(self.path) + '.lock', timeout=10)
        self.username = username

    def _read(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines() if line]

    def _write(self, rows):
        temp = self.path.with_suffix('.tmp')
        with temp.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.path)

    def pending(self):
        with self.lock:
            return self._read()

    def enqueue(self, record):
        validate(record)
        if record['username'] != self.username:
            raise ValueError('Cache user mismatch')
        with self.lock:
            rows = self._read()
            rows.append(record)
            self._write(rows)

    def drop(self, sentence_id):
        """清除作答时，把队列里还没同步的同一句也丢掉，
        否则下次 flush 会把刚删掉的记录又写回去。"""
        with self.lock:
            rows = self._read()
            kept = [r for r in rows if r['sentence_id'] != sentence_id]
            if len(kept) != len(rows):
                self._write(kept)
            return len(rows) - len(kept)

    def flush(self, store):
        with self.lock:
            rows = self._read()
            for i, row in enumerate(rows):
                try:
                    store.upsert(row)
                except Exception:
                    self._write(rows[i:])
                    return len(rows) - i
            self._write([])
            return 0

    def remember(self, rows):
        # An offline restart can recover previously synchronized answers for this user.
        if any(r['username'] != self.username for r in rows):
            raise ValueError('Snapshot user mismatch')
        snapshot = self.directory / f'{self.username}.snapshot.json'
        with self.lock:
            temp = snapshot.with_suffix('.tmp')
            with temp.open('w', encoding='utf-8') as f:
                json.dump(rows, f, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, snapshot)

    def snapshot(self):
        path = self.directory / f'{self.username}.snapshot.json'
        with self.lock:
            return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
