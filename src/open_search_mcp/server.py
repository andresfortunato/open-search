"""MCP search server: SearXNG + trafilatura + BM25."""

import asyncio
import os
import logging
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP, Context

from .searcher import search_searxng, score_with_bm25
from .cache import URLCache
from .extractor import fetch_and_extract
from .chunker import _get_model

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Configuration via environment variables
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")
FETCH_TIMEOUT = float(os.environ.get("FETCH_TIMEOUT", "4"))
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", "20000"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_FETCHES", "5"))

# docker-compose.yml lives at repo root (two levels up from this file in src layout)
COMPOSE_DIR = os.environ.get(
    "OPEN_SEARCH_COMPOSE_DIR",
    str(Path(__file__).parent.parent.parent),
)


async def _ensure_searxng_running() -> None:
    """Start SearXNG via docker compose if it's not already reachable."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as probe:
            resp = await probe.get(f"{SEARXNG_URL}/healthz")
            if resp.status_code == 200:
                return
    except Exception:
        pass

    logger.info("SearXNG not reachable, starting via docker compose...")
    await asyncio.to_thread(
        subprocess.run,
        ["docker", "compose", "up", "-d"],
        cwd=COMPOSE_DIR,
        capture_output=True,
    )

    # Wait for SearXNG to become healthy (up to 15s)
    for _ in range(15):
        await asyncio.sleep(1)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as probe:
                resp = await probe.get(f"{SEARXNG_URL}/healthz")
                if resp.status_code == 200:
                    logger.info("SearXNG is ready.")
                    return
        except Exception:
            continue
    logger.warning("SearXNG may not be fully ready after 15s — proceeding anyway.")


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Start SearXNG if needed, pre-warm embedding model, provide shared httpx client."""
    await _ensure_searxng_running()

    # Pre-warm embedding model so first search doesn't pay load penalty
    await asyncio.to_thread(_get_model)

    url_cache = URLCache(ttl_seconds=300)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(FETCH_TIMEOUT),
        headers={"User-Agent": "search-mcp/0.1 (content extraction for LLMs)"},
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    ) as client:
        yield {"http_client": client, "url_cache": url_cache}


mcp = FastMCP("search", lifespan=app_lifespan)


@mcp.tool()
async def search(
    query: str,
    max_results: int = 5,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    time_range: str | None = None,
    ctx: Context = None,
) -> str:
    """Search the web and return extracted content optimized for LLM consumption.

    Returns structured results with title, URL, relevance score, and clean
    markdown content extracted from each page.

    Args:
        query: Search query string
        max_results: Number of results to return (1-10, default 5)
        include_domains: Only include results from these domains
        exclude_domains: Exclude results from these domains
        time_range: Filter by time: 'day', 'week', 'month', 'year'
    """
    max_results = max(1, min(10, max_results))
    lifespan_ctx = ctx.request_context.lifespan_context
    client: httpx.AsyncClient = lifespan_ctx["http_client"]
    url_cache: URLCache = lifespan_ctx["url_cache"]

    # Step 1: Search SearXNG (overfetch to handle extraction failures)
    try:
        search_results = await search_searxng(
            client=client,
            query=query,
            searxng_url=SEARXNG_URL,
            max_results=max_results * 2,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            time_range=time_range,
        )
    except RuntimeError as e:
        return str(e)

    if not search_results:
        return f"No search results found for: {query}"

    urls = [r["url"] for r in search_results]

    # Step 2: Fetch and extract content concurrently (with chunk selection)
    extracted = await fetch_and_extract(
        client=client,
        urls=urls,
        query=query,
        max_results=max_results,
        cache=url_cache,
        max_concurrent=MAX_CONCURRENT,
        max_length=MAX_CONTENT_LENGTH,
    )

    # Step 3: Fall back to snippets for URLs where extraction failed
    extracted_urls = {r["url"] for r in extracted}
    for sr in search_results:
        if sr["url"] not in extracted_urls and sr["snippet"]:
            extracted.append({
                "url": sr["url"],
                "title": sr["title"],
                "content": f"[snippet] {sr['snippet']}",
            })

    if not extracted:
        return f"Found {len(search_results)} results but failed to extract content from any. Search snippets:\n\n" + "\n".join(
            f"- [{r['title']}]({r['url']}): [snippet] {r['snippet']}" for r in search_results[:max_results]
        )

    # Step 4: Score with BM25 and return top results
    scored = score_with_bm25(query, extracted, content_key="content")
    top = scored[:max_results]

    # Format as structured text for the LLM
    parts = []
    for i, r in enumerate(top, 1):
        parts.append(
            f"## Result {i} (score: {r.get('score', 'N/A')})\n"
            f"**{r['title']}**\n"
            f"{r['url']}\n\n"
            f"{r['content']}"
        )

    return "\n\n---\n\n".join(parts)


@mcp.tool()
async def extract(
    urls: str | list[str],
    query: str | None = None,
    ctx: Context = None,
) -> str:
    """Extract clean markdown content from specific URLs.

    Fetches the pages, strips boilerplate (ads, navigation, scripts), and
    returns the main content as clean markdown.

    Args:
        urls: URL or list of URLs to extract content from (max 10)
        query: Optional query to score extracted content by relevance
    """
    if isinstance(urls, str):
        urls = [urls]
    urls = urls[:10]

    lifespan_ctx = ctx.request_context.lifespan_context
    client: httpx.AsyncClient = lifespan_ctx["http_client"]
    url_cache: URLCache = lifespan_ctx["url_cache"]

    extracted = await fetch_and_extract(
        client=client,
        urls=urls,
        query=query,
        cache=url_cache,
        max_concurrent=MAX_CONCURRENT,
        max_length=MAX_CONTENT_LENGTH,
    )

    if not extracted:
        return f"Failed to extract content from any of the provided URLs: {', '.join(urls)}"

    if query:
        extracted = score_with_bm25(query, extracted, content_key="content")

    parts = []
    for r in extracted:
        score_str = f" (score: {r['score']})" if "score" in r else ""
        parts.append(
            f"## {r['title']}{score_str}\n"
            f"{r['url']}\n\n"
            f"{r['content']}"
        )

    return "\n\n---\n\n".join(parts)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
