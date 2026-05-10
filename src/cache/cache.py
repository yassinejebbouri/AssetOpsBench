from collections import OrderedDict
import json
import os
from pathlib import Path
from typing import Any

class Cache:
    def __init__(self, cache_file: str, capacity: int = 10):
        self.cache_file = Path(cache_file)
        self.cache = OrderedDict()
        self.capacity = capacity
        self._load_cache()

    def get(self, key):
        if _cache_disabled():
            return None
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def add(self, key, value):
        if _cache_disabled():
            return
        self._set(key, value)
        self._save_cache()

    def add_many(self, entries):
        if _cache_disabled():
            return
        for key, value in entries.items():
            self._set(key, value)
        self._save_cache()

    def _set(self, key, value):
        if key not in self.cache and len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = _json_safe(value)
        self.cache.move_to_end(key)

    def _load_cache(self):
        if not self.cache_file.exists():
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text("{}")
        try:
            with open(self.cache_file, "r") as f:
                data = json.load(f)
                self.cache = OrderedDict(data if isinstance(data, dict) else {})
        except json.JSONDecodeError:
            self.cache = OrderedDict()
            self._save_cache()

    def _save_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.cache_file.with_name(
            f".{self.cache_file.name}.{os.getpid()}.tmp"
        )
        with open(tmp_file, "w") as f:
            json.dump(self.cache, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, self.cache_file)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _cache_disabled() -> bool:
    return os.environ.get("ASSETOPSBENCH_DISABLE_CACHE", "").lower() in {
        "1",
        "true",
        "yes",
    }
