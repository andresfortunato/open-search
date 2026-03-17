"""Async URL fetching and content extraction via trafilatura.

Supports optional Playwright fallback for JS-rendered pages.
Install with: pip install open-search-mcp[browser]
"""

import asyncio
import logging
import random

import httpx
import trafilatura

from .cache import URLCache
from .chunker import select_chunks

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 8.0
MAX_CONCURRENT = 5
MAX_CONTENT_LENGTH = 20_000
TARGET_CHUNK_CHARS = 500

# Rotate User-Agent to reduce 403s from sites that block bots
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0",
]

# Playwright is optional — detected at import time
_playwright_available = False
try:
    from playwright.async_api import async_playwright
    _playwright_available = True
except ImportError:
    pass


async def fetch_url(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None]:
    """Fetch a single URL, returning (url, html_or_none)."""
    async with semaphore:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return (url, resp.text)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return (url, None)


async def fetch_many(
    client: httpx.AsyncClient,
    urls: list[str],
    max_concurrent: int = MAX_CONCURRENT,
) -> dict[str, str | None]:
    """Fetch multiple URLs concurrently. Returns {url: html_or_none}."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [fetch_url(client, url, semaphore) for url in urls]
    results = await asyncio.gather(*tasks)
    return dict(results)


def extract_content(
    html: str,
    url: str,
    max_length: int = MAX_CONTENT_LENGTH,
) -> dict | None:
    """Extract clean markdown content from HTML using trafilatura.

    Returns {"title": str, "content": str} or None on failure.
    """
    try:
        content = trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_links=True,
            include_tables=True,
            include_formatting=True,
        )
        if not content or len(content.strip()) < 50:
            return None

        title = None
        metadata = trafilatura.extract_metadata(html, default_url=url)
        if metadata:
            title = metadata.title

        # Truncate at paragraph boundary if too long
        if len(content) > max_length:
            cut = content[:max_length].rfind("\n\n")
            if cut > max_length // 2:
                content = content[:cut]
            else:
                content = content[:max_length]

        return {"title": title or "", "content": content.strip()}
    except Exception as e:
        logger.warning("Extraction failed for %s: %s", url, e)
        return None


async def extract_content_async(
    html: str,
    url: str,
    max_length: int = MAX_CONTENT_LENGTH,
) -> dict | None:
    """Async wrapper — runs trafilatura in a thread pool."""
    return await asyncio.to_thread(extract_content, html, url, max_length)


async def fetch_with_playwright(
    urls: list[str],
    timeout_ms: int = 8000,
) -> dict[str, str | None]:
    """Fetch URLs using a headless browser. Handles JS-rendered pages.

    Returns {url: html_or_none}. Only called for URLs that failed with httpx.
    """
    if not _playwright_available:
        return {url: None for url in urls}

    results: dict[str, str | None] = {}
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            for url in urls:
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    # Brief wait for JS rendering
                    await page.wait_for_timeout(1000)
                    html = await page.content()
                    results[url] = html
                except Exception as e:
                    logger.warning("Playwright failed for %s: %s", url, e)
                    results[url] = None
            await browser.close()
    except Exception as e:
        logger.warning("Playwright browser launch failed: %s", e)
        return {url: None for url in urls}
    return results


async def fetch_and_extract(
    client: httpx.AsyncClient,
    urls: list[str],
    query: str | None = None,
    max_results: int | None = None,
    cache: URLCache | None = None,
    max_concurrent: int = MAX_CONCURRENT,
    max_length: int = MAX_CONTENT_LENGTH,
    target_chars: int = TARGET_CHUNK_CHARS,
) -> list[dict]:
    """Fetch URLs and extract content concurrently with early return.

    Returns list of {"url": str, "title": str, "content": str}.
    Processes pages as fetches complete. Stops collecting once max_results
    pages are successfully extracted. When query is provided, content is
    reduced to the most query-relevant chunks via embeddings.
    Optionally caches extracted content (pre-chunking) by URL.
    """
    target_count = max_results if max_results else len(urls)

    # Separate cached and uncached URLs
    results = []
    urls_to_fetch = []
    for url in urls:
        if cache:
            cached = cache.get(url)
            if cached:
                results.append({"url": url, **cached})
                if len(results) >= target_count:
                    break
                continue
        urls_to_fetch.append(url)

    # If cache satisfied everything, skip fetching
    if len(results) >= target_count:
        # Still apply chunk selection
        return await _apply_chunk_selection(results, query, target_chars)

    semaphore = asyncio.Semaphore(max_concurrent)

    # Launch all fetches as tasks
    async def fetch_and_process(url: str) -> dict | None:
        """Fetch one URL, extract content. Returns result dict or None."""
        async with semaphore:
            try:
                headers = {"User-Agent": random.choice(_USER_AGENTS)}
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                html = resp.text
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", url, e)
                return None

        extracted = await extract_content_async(html, url, max_length)
        if not extracted:
            return None
        return {"url": url, **extracted}

    # Create all tasks and process as they complete
    tasks = {asyncio.ensure_future(fetch_and_process(url)): url for url in urls_to_fetch}
    failed_urls = []

    for coro in asyncio.as_completed(tasks.keys()):
        result = await coro
        if result:
            # Cache the pre-chunking content
            if cache:
                cache.put(result["url"], {"title": result["title"], "content": result["content"]})
            results.append(result)
            if len(results) >= target_count:
                break
        else:
            # Find which URL this task was for
            for task, url in tasks.items():
                if task.done() and task.result() is None:
                    if url not in [r["url"] for r in results] and url not in failed_urls:
                        failed_urls.append(url)

    # Cancel remaining tasks if we got enough
    for task in tasks:
        if not task.done():
            task.cancel()
            # Track cancelled URLs as failed
            url = tasks[task]
            if url not in failed_urls:
                failed_urls.append(url)

    # Playwright fallback — only if we don't have enough results
    if len(results) < target_count and failed_urls and _playwright_available:
        needed = target_count - len(results)
        retry_urls = failed_urls[:needed * 2]  # try up to 2x what we need
        logger.info("Retrying %d URLs with Playwright...", len(retry_urls))
        pw_html = await fetch_with_playwright(retry_urls)
        for url, html in pw_html.items():
            if html and len(results) < target_count:
                extracted = await extract_content_async(html, url, max_length)
                if extracted:
                    results.append({"url": url, **extracted})

    return await _apply_chunk_selection(results, query, target_chars)


async def _apply_chunk_selection(
    results: list[dict],
    query: str | None,
    target_chars: int,
) -> list[dict]:
    """Apply embeddings-based chunk selection to all results."""
    final = []
    for r in results:
        content = r["content"]
        if query and len(content) > target_chars:
            content = await asyncio.to_thread(
                select_chunks, query, content, target_chars
            )
        final.append({
            "url": r["url"],
            "title": r["title"],
            "content": content,
        })
    return final
