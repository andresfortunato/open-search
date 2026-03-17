"""TTL-based URL cache for extracted content.

Caches post-trafilatura, pre-chunking content keyed by URL.
Chunk selection still runs on cache hits since the query may differ.
"""

import copy
import time


MAX_CACHE_ENTRIES = 500


class URLCache:
    """Simple in-memory TTL cache for extracted page content."""

    def __init__(self, ttl_seconds: int = 300, max_entries: int = MAX_CACHE_ENTRIES):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: dict[str, dict] = {}

    def get(self, url: str) -> dict | None:
        """Return cached content for URL, or None if missing/expired."""
        entry = self._entries.get(url)
        if entry is None:
            return None

        if time.monotonic() - entry["ts"] > self.ttl_seconds:
            del self._entries[url]
            return None

        return copy.deepcopy(entry["data"])

    def put(self, url: str, data: dict) -> None:
        """Cache extracted content for a URL. Evicts oldest entry if at capacity."""
        if len(self._entries) >= self.max_entries and url not in self._entries:
            oldest = min(self._entries, key=lambda k: self._entries[k]["ts"])
            del self._entries[oldest]
        self._entries[url] = {
            "ts": time.monotonic(),
            "data": copy.deepcopy(data),
        }

    def clear(self) -> None:
        """Remove all cached entries."""
        self._entries.clear()
