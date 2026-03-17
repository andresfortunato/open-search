"""Tests for extractor module — stream processing, early return, and caching."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from open_search_mcp.cache import URLCache

import httpx
import pytest


@pytest.fixture
def mock_client():
    """Create a mock httpx client."""
    return AsyncMock(spec=httpx.AsyncClient)


def _make_html(title: str, body: str) -> str:
    """Create minimal HTML that trafilatura can extract from."""
    return f"<html><head><title>{title}</title></head><body><article><p>{body}</p></article></body></html>"


# --- Item 1: Stream processing / early return ---

@pytest.mark.asyncio
async def test_early_return_when_enough_results(mock_client):
    """fetch_and_extract returns early once max_results pages are extracted,
    without waiting for all URLs to complete."""
    from open_search_mcp.extractor import fetch_and_extract

    urls = [f"https://example.com/page{i}" for i in range(10)]
    fast_content = "This is substantial content for testing extraction quality. " * 5

    call_count = 0

    async def mock_get(url):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.status_code = 200
        resp.text = _make_html(f"Page {call_count}", fast_content)
        resp.raise_for_status = MagicMock()
        return resp

    mock_client.get = mock_get

    results = await fetch_and_extract(
        client=mock_client,
        urls=urls,
        max_results=3,
    )

    # Should have exactly max_results (3) results, not all 10
    assert len(results) == 3
    for r in results:
        assert "url" in r
        assert "title" in r
        assert "content" in r


@pytest.mark.asyncio
async def test_returns_fewer_when_not_enough_succeed(mock_client):
    """When fewer than max_results URLs succeed, return what we have."""
    from open_search_mcp.extractor import fetch_and_extract

    urls = [f"https://example.com/page{i}" for i in range(5)]
    content = "This is substantial content for testing extraction quality. " * 5

    async def mock_get(url):
        resp = MagicMock()
        if "page0" in url or "page1" in url:
            resp.status_code = 200
            resp.text = _make_html("Good Page", content)
            resp.raise_for_status = MagicMock()
        else:
            resp.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
                "403", request=MagicMock(), response=MagicMock()))
        return resp

    mock_client.get = mock_get

    with patch("open_search_mcp.extractor._playwright_available", False):
        results = await fetch_and_extract(
            client=mock_client,
            urls=urls,
            max_results=5,
        )

    # Only 2 URLs succeeded, so we get 2 results
    assert len(results) == 2


@pytest.mark.asyncio
async def test_all_urls_fail_returns_empty(mock_client):
    """When all URLs fail, return empty list."""
    from open_search_mcp.extractor import fetch_and_extract

    urls = ["https://example.com/bad1", "https://example.com/bad2"]

    async def mock_get(url):
        raise httpx.ConnectError("Connection refused")

    mock_client.get = mock_get

    with patch("open_search_mcp.extractor._playwright_available", False):
        results = await fetch_and_extract(
            client=mock_client,
            urls=urls,
            max_results=5,
        )

    assert results == []


@pytest.mark.asyncio
async def test_skips_playwright_when_enough_results(mock_client):
    """Playwright fallback is NOT called when we already have enough results."""
    from open_search_mcp.extractor import fetch_and_extract

    urls = [f"https://example.com/page{i}" for i in range(6)]
    content = "This is substantial content for testing extraction quality. " * 5

    async def mock_get(url):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = _make_html("Good Page", content)
        resp.raise_for_status = MagicMock()
        return resp

    mock_client.get = mock_get

    with patch("open_search_mcp.extractor._playwright_available", True), \
         patch("open_search_mcp.extractor.fetch_with_playwright") as pw_mock:
        results = await fetch_and_extract(
            client=mock_client,
            urls=urls,
            max_results=3,
        )

    # Should have 3 results and Playwright should NOT have been called
    assert len(results) == 3
    pw_mock.assert_not_called()


@pytest.mark.asyncio
async def test_chunk_selection_applied_with_query(mock_client):
    """When query is provided, chunk selection reduces content."""
    from open_search_mcp.extractor import fetch_and_extract

    urls = ["https://example.com/page1"]
    # Content long enough to trigger chunk selection (>500 chars)
    long_content = ("Python is a great programming language for web development. " * 30 +
                    "Rate limiting protects APIs from abuse. " * 20)

    async def mock_get(url):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = _make_html("Rate Limiting Guide", long_content)
        resp.raise_for_status = MagicMock()
        return resp

    mock_client.get = mock_get

    results = await fetch_and_extract(
        client=mock_client,
        urls=urls,
        query="rate limiting",
        max_results=5,
    )

    assert len(results) == 1
    # Content should be chunked down from the full page
    assert len(results[0]["content"]) < len(long_content)


# --- Item 2: Cache integration ---

@pytest.mark.asyncio
async def test_cache_hit_avoids_refetch(mock_client):
    """When a URL is cached, fetch_and_extract uses cache and skips HTTP fetch."""
    from open_search_mcp.extractor import fetch_and_extract

    cache = URLCache(ttl_seconds=300)
    cache.put("https://example.com/cached", {"title": "Cached Title", "content": "Cached body text"})

    fetch_count = 0

    async def mock_get(url):
        nonlocal fetch_count
        fetch_count += 1
        resp = MagicMock()
        resp.status_code = 200
        content = "Fresh content from the web. " * 5
        resp.text = _make_html("Fresh Title", content)
        resp.raise_for_status = MagicMock()
        return resp

    mock_client.get = mock_get

    results = await fetch_and_extract(
        client=mock_client,
        urls=["https://example.com/cached", "https://example.com/fresh"],
        cache=cache,
        max_results=5,
    )

    assert len(results) == 2
    # Cached URL should not trigger a fetch
    assert fetch_count == 1  # only the fresh URL was fetched
    # Cached result should have cached content
    cached_result = [r for r in results if r["url"] == "https://example.com/cached"][0]
    assert cached_result["title"] == "Cached Title"


@pytest.mark.asyncio
async def test_fetched_urls_get_cached(mock_client):
    """Successfully fetched URLs should be added to the cache."""
    from open_search_mcp.extractor import fetch_and_extract

    cache = URLCache(ttl_seconds=300)
    content = "This is substantial content for testing extraction quality. " * 5

    async def mock_get(url):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = _make_html("Page Title", content)
        resp.raise_for_status = MagicMock()
        return resp

    mock_client.get = mock_get

    await fetch_and_extract(
        client=mock_client,
        urls=["https://example.com/page1"],
        cache=cache,
        max_results=5,
    )

    # URL should now be cached
    cached = cache.get("https://example.com/page1")
    assert cached is not None
    assert cached["title"] == "Page Title"
