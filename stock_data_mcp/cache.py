import sys
import time
import pathlib
import threading
import diskcache
from cachetools import TTLCache


class CacheKey:
    ALL: dict = {}
    MAX_CACHE_ENTRIES = 500  # 最大缓存条目数
    _last_cleanup = 0
    _cleanup_interval = 3600  # 每小时清理一次
    _lock = threading.Lock()  # 类级别锁，保护 ALL 字典

    def __init__(self, key, ttl=600, ttl2=None, maxsize=100):
        self.key = key
        self.ttl = ttl
        self.ttl2 = ttl2 or (ttl * 2)
        self.cache1 = TTLCache(maxsize=maxsize, ttl=ttl)
        self.cache2 = diskcache.Cache(self.get_cache_dir())
        self._created_at = time.time()
        self._instance_lock = threading.Lock()  # 实例级别锁，保护 cache1/cache2

    @staticmethod
    def init(key, ttl=600, ttl2=None, maxsize=100):
        # 定期清理过期缓存
        CacheKey._maybe_cleanup()

        with CacheKey._lock:
            if key in CacheKey.ALL:
                return CacheKey.ALL[key]
            cache = CacheKey(key, ttl, ttl2, maxsize)
            return CacheKey.ALL.setdefault(key, cache)

    @staticmethod
    def _maybe_cleanup():
        """定期清理过期缓存"""
        now = time.time()
        if now - CacheKey._last_cleanup < CacheKey._cleanup_interval:
            return

        with CacheKey._lock:
            # 再次检查（双重检查锁定）
            if now - CacheKey._last_cleanup < CacheKey._cleanup_interval:
                return
            CacheKey._last_cleanup = now

            # 如果缓存条目超过限制，清理最旧的条目
            if len(CacheKey.ALL) > CacheKey.MAX_CACHE_ENTRIES:
                # 按创建时间排序，删除最旧的一半
                sorted_keys = sorted(
                    CacheKey.ALL.keys(),
                    key=lambda k: CacheKey.ALL[k]._created_at
                )
                for key in sorted_keys[:len(sorted_keys) // 2]:
                    try:
                        CacheKey.ALL[key].cache2.close()
                    except Exception:  # 忽略缓存关闭时的异常
                        pass
                    del CacheKey.ALL[key]

    @staticmethod
    def clear_all():
        """清理所有缓存"""
        with CacheKey._lock:
            for cache in CacheKey.ALL.values():
                try:
                    cache.cache1.clear()
                    cache.cache2.close()
                except Exception:  # 忽略缓存清理时的异常
                    pass
            CacheKey.ALL.clear()

    def get(self):
        with self._instance_lock:
            try:
                return self.cache1[self.key]
            except KeyError:
                pass
            return self.cache2.get(self.key)

    def set(self, val):
        with self._instance_lock:
            self.cache1[self.key] = val
            self.cache2.set(self.key, val, expire=self.ttl2)
            return val

    def delete(self):
        with self._instance_lock:
            self.cache1.pop(self.key, None)
            self.cache2.delete(self.key)

    def get_cache_dir(self):
        home = pathlib.Path.home()
        name = __package__
        if sys.platform == "win32":
            return home / "AppData" / "Local" / "Cache" / name
        return home / ".cache" / name
