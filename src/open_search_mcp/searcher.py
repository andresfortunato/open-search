"""SearXNG client and BM25 relevance scoring."""

import logging
from urllib.parse import urlparse

import httpx
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


async def search_searxng(
    client: httpx.AsyncClient,
    query: str,
    searxng_url: str,
    max_results: int = 10,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    time_range: str | None = None,
) -> list[dict]:
    """Query SearXNG and return deduplicated results.

    Returns list of {"title": str, "url": str, "snippet": str, "engines": list[str]}.
    """
    params: dict = {
        "q": query,
        "format": "json",
        "pageno": 1,
    }
    if time_range and time_range in ("day", "week", "month", "year"):
        params["time_range"] = time_range

    try:
        resp = await client.get(f"{searxng_url}/search", params=params)
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        raise RuntimeError(
            f"SearXNG is not reachable at {searxng_url}. "
            "Is Docker Compose running? (docker compose up -d)"
        )
    except Exception as e:
        raise RuntimeError(f"SearXNG query failed: {e}")

    raw_results = data.get("results", [])

    # Deduplicate by URL
    seen_urls: set[str] = set()
    results: list[dict] = []
    for r in raw_results:
        url = r.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        hostname = urlparse(url).hostname or ""

        # Domain filtering
        if include_domains:
            if not any(hostname.endswith(d) for d in include_domains):
                continue
        if exclude_domains:
            if any(hostname.endswith(d) for d in exclude_domains):
                continue

        results.append({
            "title": r.get("title", ""),
            "url": url,
            "snippet": r.get("content", ""),
            "engines": r.get("engines", []),
        })

        if len(results) >= max_results:
            break

    return results


def score_with_bm25(
    query: str,
    documents: list[dict],
    content_key: str = "content",
) -> list[dict]:
    """Score documents against query using BM25. Adds normalized 'score' field.

    Returns documents sorted by score descending.
    """
    if not documents:
        return documents

    # Simple whitespace tokenization (sufficient for BM25 on small corpus)
    tokenized_query = query.lower().split()
    corpus = [doc.get(content_key, "").lower().split() for doc in documents]

    # Handle edge case: all empty documents
    if all(len(tokens) == 0 for tokens in corpus):
        for doc in documents:
            doc["score"] = 0.0
        return documents

    bm25 = BM25Okapi(corpus)
    raw_scores = bm25.get_scores(tokenized_query)

    # Normalize to 0.0-1.0
    min_score = float(min(raw_scores))
    max_score = float(max(raw_scores))
    score_range = max_score - min_score

    for doc, raw in zip(documents, raw_scores):
        if score_range > 0:
            doc["score"] = round((float(raw) - min_score) / score_range, 4)
        else:
            doc["score"] = 1.0  # All scores equal → all equally relevant

    documents.sort(key=lambda d: d["score"], reverse=True)
    return documents
