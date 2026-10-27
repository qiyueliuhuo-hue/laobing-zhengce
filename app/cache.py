"""两级缓存：L1 进程内存 + L2 sqlite 落盘。

用法：
    from cache import cache
    cache.set("key", {"data": ...})
    hit = cache.get("key")   # None = miss

线程安全：L1 用锁保护；L2 每次短连接 sqlite（WAL 关闭、单写），
并发下由 sqlite 自带锁串行化，简单可靠。
"""
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict

_CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
_L2_PATH = os.path.join(_CACHE_DIR, "l2_cache.sqlite")

# L1 内存上限（条数），超出逐出最旧
L1_MAX = 500
# 默认有效期：None = 不过期（政策库重建/更新后调用 cache.clear() 手动清）
DEFAULT_TTL = None


class _TieredCache:
    def __init__(self, l2_path: str = _L2_PATH):
        self._l1 = OrderedDict()  # key -> {"v": value, "t": expire_ts or None}
        self._l1_lock = threading.Lock()
        self._l2_path = l2_path
        self._ensure_l2()

    # ---------- L2 基础 ----------
    def _ensure_l2(self):
        try:
            with sqlite3.connect(self._l2_path, timeout=5) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS cache ("
                    "  k TEXT PRIMARY KEY,"
                    "  v TEXT NOT NULL,"
                    "  expire_ts INTEGER"
                    ")"
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expire ON cache(expire_ts)")
        except Exception as e:
            # L2 不可用（磁盘/权限）不致命：缓存退化为仅 L1
            print(f"[cache][warn] L2 sqlite 不可用，仅用内存缓存: {e}")

    def _l2_get(self, key: str):
        try:
            with sqlite3.connect(self._l2_path, timeout=5) as conn:
                row = conn.execute(
                    "SELECT v, expire_ts FROM cache WHERE k=?", (key,)
                ).fetchone()
            if not row:
                return None
            v, expire_ts = row
            if expire_ts is not None and expire_ts < int(time.time()):
                return None  # 过期视为 miss（惰性删除）
            return json.loads(v)
        except Exception:
            return None

    def _l2_set(self, key: str, value, ttl):
        try:
            expire_ts = None if ttl is None else int(time.time()) + ttl
            with sqlite3.connect(self._l2_path, timeout=5) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (k, v, expire_ts) VALUES (?,?,?)",
                    (key, json.dumps(value, ensure_ascii=False), expire_ts),
                )
        except Exception as e:
            print(f"[cache][warn] L2 写入失败: {e}")

    # ---------- 对外接口 ----------
    def get(self, key: str):
        # L1
        with self._l1_lock:
            item = self._l1.get(key)
            if item is not None:
                if item["t"] is None or item["t"] > int(time.time()):
                    self._l1.move_to_end(key)  # LRU 刷新
                    return item["v"]
                else:
                    self._l1.pop(key, None)
        # L2
        val = self._l2_get(key)
        if val is not None:
            with self._l1_lock:
                if len(self._l1) >= L1_MAX:
                    self._l1.popitem(last=False)
                self._l1[key] = {"v": val, "t": None}
        return val

    def set(self, key: str, value, ttl=DEFAULT_TTL):
        expire_ts = None if ttl is None else int(time.time()) + ttl
        with self._l1_lock:
            if len(self._l1) >= L1_MAX:
                self._l1.popitem(last=False)
            self._l1[key] = {"v": value, "t": expire_ts}
        self._l2_set(key, value, ttl)

    def clear(self):
        """清空两级缓存（政策库重建/更新后调用）"""
        with self._l1_lock:
            self._l1.clear()
        try:
            with sqlite3.connect(self._l2_path, timeout=5) as conn:
                conn.execute("DELETE FROM cache")
        except Exception:
            pass

    def stats(self) -> dict:
        with self._l1_lock:
            l1 = len(self._l1)
        try:
            with sqlite3.connect(self._l2_path, timeout=5) as conn:
                l2 = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        except Exception:
            l2 = -1
        return {"l1_items": l1, "l2_items": l2}


cache = _TieredCache()
