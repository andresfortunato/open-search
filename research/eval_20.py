"""Eval: Run 20 diverse queries through our tool, measure latency + tokens."""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_search_mcp.searcher import search_searxng, score_with_bm25
from open_search_mcp.extractor import fetch_and_extract
from open_search_mcp.chunker import _get_model

SEARXNG_URL = "http://localhost:8888"
MAX_RESULTS = 5

QUERIES = [
    # Technical / programming
    "how to implement rate limiting in FastAPI",
    "rust async await best practices",
    "kubernetes horizontal pod autoscaler configuration",
    "PostgreSQL window functions examples",
    "WebSocket vs Server-Sent Events comparison",
    # Science / engineering
    "what causes lithium battery thermal runaway",
    "how does mRNA vaccine technology work",
    "CRISPR gene editing mechanism explained",
    "quantum computing error correction methods",
    "how do solar panels convert light to electricity",
    # Practical / how-to
    "how to set up a SearXNG instance",
    "best practices for Docker container security",
    "how to configure nginx reverse proxy",
    "git rebase vs merge when to use which",
    "how to optimize Python memory usage",
    # Domain-specific
    "GDPR data processing agreement requirements",
    "transformer architecture attention mechanism explained",
    "climate change impact on coral reef ecosystems",
    "microservices vs monolith architecture tradeoffs",
    "zero trust network security architecture",
]


async def run_query(client, query):
    t0 = time.perf_counter()
    results = await search_searxng(client, query, SEARXNG_URL, max_results=MAX_RESULTS * 2)
    urls = [r["url"] for r in results]
    extracted = await fetch_and_extract(client, urls, query=query, max_results=MAX_RESULTS)

    # Snippet fallback
    extracted_urls = {r["url"] for r in extracted}
    for sr in results:
        if sr["url"] not in extracted_urls and sr["snippet"]:
            extracted.append({"url": sr["url"], "title": sr["title"], "content": f"[snippet] {sr['snippet']}"})

    scored = score_with_bm25(query, extracted, content_key="content")
    top = scored[:MAX_RESULTS]
    total_ms = (time.perf_counter() - t0) * 1000

    parts = []
    for i, r in enumerate(top, 1):
        parts.append(f"## Result {i}\n**{r['title']}**\n{r['url']}\n\n{r['content']}")
    output = "\n\n---\n\n".join(parts)

    return {
        "query": query,
        "total_ms": round(total_ms),
        "num_results": len(top),
        "total_chars": len(output),
        "est_tokens": round(len(output) / 4),
    }


async def main():
    print("Pre-warming model...")
    _get_model()

    results = []
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(4.0),
        headers={"User-Agent": "search-mcp/0.1"},
    ) as client:
        for qi, query in enumerate(QUERIES, 1):
            r = await run_query(client, query)
            results.append(r)
            print(f"[{qi:>2}/{len(QUERIES)}] {r['total_ms']:>5}ms {r['est_tokens']:>4}tok {r['num_results']}r  {query[:55]}")

    out = Path(__file__).parent / "eval_20_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    avg_ms = sum(r["total_ms"] for r in results) / len(results)
    avg_tok = sum(r["est_tokens"] for r in results) / len(results)
    avg_results = sum(r["num_results"] for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"20 queries: avg {avg_ms:.0f}ms, ~{avg_tok:.0f}tok/query, {avg_results:.1f} results/query")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
