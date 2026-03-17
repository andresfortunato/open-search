"""Tests for server module — SearXNG error recovery."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_search_retries_after_searxng_crash():
    """When SearXNG is unreachable, search retries after restarting it."""
    from open_search_mcp.server import _search_with_recovery

    call_count = 0

    async def mock_search_searxng(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("SearXNG is not reachable")
        # Second call succeeds
        return [
            {"title": "Result 1", "url": "https://example.com/1", "snippet": "Snippet 1", "engines": ["google"]},
        ]

    with patch("open_search_mcp.server.search_searxng", side_effect=mock_search_searxng), \
         patch("open_search_mcp.server._ensure_searxng_running", new_callable=AsyncMock) as mock_restart:
        results = await _search_with_recovery(
            client=AsyncMock(),
            query="test query",
            searxng_url="http://localhost:8888",
            max_results=5,
        )

    assert mock_restart.call_count == 1
    assert len(results) == 1
    assert results[0]["title"] == "Result 1"


@pytest.mark.asyncio
async def test_search_returns_error_after_failed_retry():
    """When SearXNG restart fails, search returns the error."""
    from open_search_mcp.server import _search_with_recovery

    async def mock_search_searxng(**kwargs):
        raise RuntimeError("SearXNG is not reachable at http://localhost:8888")

    with patch("open_search_mcp.server.search_searxng", side_effect=mock_search_searxng), \
         patch("open_search_mcp.server._ensure_searxng_running", new_callable=AsyncMock):
        with pytest.raises(RuntimeError, match="SearXNG is not reachable"):
            await _search_with_recovery(
                client=AsyncMock(),
                query="test query",
                searxng_url="http://localhost:8888",
                max_results=5,
            )


@pytest.mark.asyncio
async def test_search_no_retry_on_non_connection_error():
    """Non-connection RuntimeErrors (e.g. query parse error) are not retried."""
    from open_search_mcp.server import _search_with_recovery

    async def mock_search_searxng(**kwargs):
        raise RuntimeError("SearXNG query failed: bad request")

    with patch("open_search_mcp.server.search_searxng", side_effect=mock_search_searxng), \
         patch("open_search_mcp.server._ensure_searxng_running", new_callable=AsyncMock) as mock_restart:
        with pytest.raises(RuntimeError, match="bad request"):
            await _search_with_recovery(
                client=AsyncMock(),
                query="test query",
                searxng_url="http://localhost:8888",
                max_results=5,
            )

    # Should NOT have attempted a restart
    mock_restart.assert_not_called()
