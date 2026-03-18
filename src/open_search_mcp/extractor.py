"""Async URL fetching and content extraction via trafilatura.

Uses Playwright (headless Chromium) as the primary fetcher for fully-rendered
DOM content. Falls back to httpx for speed when Playwright is unavailable.
"""

import asyncio
import ipaddress
import logging
import random
from urllib.parse import urlparse

import httpx
import trafilatura

from .cache import URLCache
from .chunker import select_chunks

logger = logging.getLogger(__name__)

# Max bytes to read from a single HTTP response (prevents memory exhaustion)
MAX_FETCH_BYTES = 5 * 1024 * 1024  # 5 MB

_PRIVATE_NETS = [
    ipaddress.ip_network(n) for n in (
        "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
        "127.0.0.0/8", "169.254.0.0/16", "::1/128", "fc00::/7",
    )
]


def _validate_url(url: str) -> None:
    """Reject URLs with non-HTTP schemes or private/internal IP addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Disallowed URL scheme: {parsed.scheme!r}")
    host = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(host)
        if any(addr in net for net in _PRIVATE_NETS):
            raise ValueError(f"Blocked private address: {host!r}")
    except ValueError as e:
        if "Blocked" in str(e):
            raise


# Rotate User-Agent to reduce 403s
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0",
]

from playwright.async_api import async_playwright


def extract_content(
    html: str,
    url: str,
    max_length: int = 20_000,
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
        logger.warning("Extraction failed for %r: %s", url, e)
        return None


async def extract_content_async(
    html: str,
    url: str,
    max_length: int = 20_000,
) -> dict | None:
    """Async wrapper — runs trafilatura in a thread pool."""
    return await asyncio.to_thread(extract_content, html, url, max_length)


class PlaywrightBrowser:
    """Persistent headless browser for page fetching.

    Reuses a single Chromium instance across searches. Each URL gets a fresh
    browser context (isolated cookies/storage) to prevent cross-site leakage.
    Fetches run in parallel using asyncio.gather.
    """

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None

    async def start(self) -> None:
        """Launch the browser. Called once during MCP lifespan startup."""
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            logger.info("Playwright browser launched.")
        except Exception as e:
            logger.warning(
                "Playwright browser not available: %s. "
                "Run 'playwright install chromium' for better extraction.",
                e,
            )
            if self._playwright:
                await self._playwright.stop()
            self._playwright = None
            self._browser = None

    async def stop(self) -> None:
        """Close the browser. Called during MCP lifespan shutdown."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser = None
        self._playwright = None

    @property
    def available(self) -> bool:
        return self._browser is not None

    async def _fetch_one(self, url: str, timeout_ms: int) -> tuple[str, str | None]:
        """Fetch a single URL in its own browser context."""
        try:
            _validate_url(url)
        except ValueError:
            return (url, None)

        assert self._browser is not None
        context = await self._browser.new_context(
            user_agent=random.choice(_USER_AGENTS),
        )
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            await page.wait_for_timeout(1500)
            html = await page.content()
            return (url, html)
        except Exception as e:
            logger.warning("Playwright failed for %r: %s", url, e)
            return (url, None)
        finally:
            await context.close()

    async def fetch_parallel(
        self,
        urls: list[str],
        timeout_ms: int = 10000,
    ) -> dict[str, str | None]:
        """Fetch multiple URLs in parallel using separate browser contexts."""
        if not self.available:
            return {url: None for url in urls}

        tasks = [self._fetch_one(url, timeout_ms) for url in urls]
        results = await asyncio.gather(*tasks)
        return dict(results)


async def _fetch_with_httpx(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None]:
    """Fetch a single URL with httpx (fast path, no JS rendering)."""
    try:
        _validate_url(url)
    except ValueError:
        return (url, None)

    async with semaphore:
        try:
            headers = {"User-Agent": random.choice(_USER_AGENTS)}
            resp = await client.get(url, headers=headers)

            # Follow redirects manually with SSRF validation (max 5 hops)
            for _ in range(5):
                if resp.status_code not in (301, 302, 303, 307, 308):
                    break
                redirect_url = resp.headers.get("location", "")
                if not redirect_url:
                    break
                if redirect_url.startswith("/"):
                    parsed = urlparse(url)
                    redirect_url = f"{parsed.scheme}://{parsed.netloc}{redirect_url}"
                _validate_url(redirect_url)
                resp = await client.get(redirect_url, headers=headers)

            resp.raise_for_status()
            content_length = resp.headers.get("content-length")
            if content_length and int(content_length) > MAX_FETCH_BYTES:
                return (url, None)
            return (url, resp.text[:MAX_FETCH_BYTES])
        except Exception as e:
            logger.warning("httpx failed for %r: %s", url, e)
            return (url, None)


async def fetch_and_extract(
    client: httpx.AsyncClient,
    urls: list[str],
    query: str | None = None,
    max_results: int | None = None,
    cache: URLCache | None = None,
    browser: PlaywrightBrowser | None = None,
    max_concurrent: int = 5,
    max_length: int = 20_000,
    target_chars: int = 1500,
) -> list[dict]:
    """Fetch URLs and extract content. Playwright-first when available.

    Strategy:
    - If Playwright is available: fetch all URLs in parallel via browser
    - If not: fetch with httpx (fast but misses JS-rendered pages)
    - Cache results for reuse across searches
    - Apply embeddings-based chunk selection when query is provided
    """
    target_count = max_results if max_results else len(urls)

    # Check cache first
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

    if len(results) >= target_count:
        return await _apply_chunk_selection(results, query, target_chars)

    # Fetch pages with httpx (fast path)
    semaphore = asyncio.Semaphore(max_concurrent)
    fetch_tasks = {
        asyncio.ensure_future(_fetch_with_httpx(client, url, semaphore)): url
        for url in urls_to_fetch
    }
    failed_urls = []

    for coro in asyncio.as_completed(fetch_tasks.keys()):
        url, html = await coro
        if html:
            extracted = await extract_content_async(html, url, max_length)
            if extracted:
                if cache:
                    cache.put(url, {"title": extracted["title"], "content": extracted["content"]})
                results.append({"url": url, **extracted})
                if len(results) >= target_count:
                    break
            else:
                failed_urls.append(url)
        else:
            failed_urls.append(url)

    # Cancel remaining httpx tasks if we got enough
    for task in fetch_tasks:
        if not task.done():
            task.cancel()
            url = fetch_tasks[task]
            if url not in failed_urls:
                failed_urls.append(url)

    # Playwright fallback for failed URLs (JS-rendered pages, 403s)
    if len(results) < target_count and failed_urls and browser and browser.available:
        needed = target_count - len(results)
        retry_urls = failed_urls[:needed * 2]
        logger.info("Retrying %d URLs with Playwright...", len(retry_urls))
        pw_html = await browser.fetch_parallel(retry_urls)
        for url in retry_urls:
            if len(results) >= target_count:
                break
            html = pw_html.get(url)
            if html:
                extracted = await extract_content_async(html, url, max_length)
                if extracted:
                    if cache:
                        cache.put(url, {"title": extracted["title"], "content": extracted["content"]})
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
