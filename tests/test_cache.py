"""Tests for URL caching layer."""

import time
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_cache_hit_skips_fetch():
    """Cached URL returns stored content without fetching."""
    from open_search_mcp.cache import URLCache

    cache = URLCache(ttl_seconds=300)
    cache.put("https://example.com/page1", {"title": "Test", "content": "Cached content here"})

    result = cache.get("https://example.com/page1")
    assert result is not None
    assert result["title"] == "Test"
    assert result["content"] == "Cached content here"


@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    """Uncached URL returns None."""
    from open_search_mcp.cache import URLCache

    cache = URLCache(ttl_seconds=300)
    result = cache.get("https://example.com/not-cached")
    assert result is None


@pytest.mark.asyncio
async def test_cache_ttl_expiry():
    """Entries older than TTL are evicted."""
    from open_search_mcp.cache import URLCache

    cache = URLCache(ttl_seconds=1)
    cache.put("https://example.com/page1", {"title": "Test", "content": "Content"})

    # Still valid immediately
    assert cache.get("https://example.com/page1") is not None

    # Expire it by manipulating the timestamp
    cache._entries["https://example.com/page1"]["ts"] = time.monotonic() - 2

    # Now expired
    assert cache.get("https://example.com/page1") is None


@pytest.mark.asyncio
async def test_cache_clear():
    """Cache can be cleared."""
    from open_search_mcp.cache import URLCache

    cache = URLCache(ttl_seconds=300)
    cache.put("https://example.com/page1", {"title": "T", "content": "C"})
    cache.put("https://example.com/page2", {"title": "T2", "content": "C2"})

    assert cache.get("https://example.com/page1") is not None
    cache.clear()
    assert cache.get("https://example.com/page1") is None
    assert cache.get("https://example.com/page2") is None


@pytest.mark.asyncio
async def test_cache_returns_copy_not_reference():
    """Cached data should be a copy so modifications don't affect the cache."""
    from open_search_mcp.cache import URLCache

    cache = URLCache(ttl_seconds=300)
    cache.put("https://example.com/page1", {"title": "Test", "content": "Original"})

    result = cache.get("https://example.com/page1")
    result["content"] = "Modified"

    # Original cache entry should be unchanged
    result2 = cache.get("https://example.com/page1")
    assert result2["content"] == "Original"
