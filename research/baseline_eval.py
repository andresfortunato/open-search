"""Baseline eval: Run our tool on 5 queries, capture output + timing.

Saves full outputs for quality comparison against WebSearch+WebFetch.
"""

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

# 5 diverse queries for the eval
QUERIES = [
    "how to implement rate limiting in FastAPI",
    "what causes lithium battery thermal runaway",
    "best practices for PostgreSQL index optimization",
    "how to set up a SearXNG instance",
    "transformer architecture attention mechanism explained",
]


async def run_query(client: httpx.AsyncClient, query: str) -> dict:
    """Run full pipeline for one query, return metrics + output."""
    t0 = time.perf_counter()

    # Search
    t_search = time.perf_counter()
    results = await search_searxng(client, query, SEARXNG_URL, max_results=MAX_RESULTS * 2)
    search_ms = (time.perf_counter() - t_search) * 1000

    urls = [r["url"] for r in results]

    # Fetch + extract + chunk
    t_extract = time.perf_counter()
    extracted = await fetch_and_extract(client, urls, query=query, max_results=MAX_RESULTS)
    extract_ms = (time.perf_counter() - t_extract) * 1000

    # Snippet fallback
    extracted_urls = {r["url"] for r in extracted}
    for sr in results:
        if sr["url"] not in extracted_urls and sr["snippet"]:
            extracted.append({
                "url": sr["url"],
                "title": sr["title"],
                "content": f"[snippet] {sr['snippet']}",
            })

    # BM25 rank
    scored = score_with_bm25(query, extracted, content_key="content")
    top = scored[:MAX_RESULTS]

    total_ms = (time.perf_counter() - t0) * 1000

    # Format output exactly as the MCP tool would
    parts = []
    for i, r in enumerate(top, 1):
        parts.append(
            f"## Result {i} (score: {r.get('score', 'N/A')})\n"
            f"**{r['title']}**\n"
            f"{r['url']}\n\n"
            f"{r['content']}"
        )
    output = "\n\n---\n\n".join(parts)

    return {
        "query": query,
        "output": output,
        "total_ms": round(total_ms),
        "search_ms": round(search_ms),
        "extract_ms": round(extract_ms),
        "num_results": len(top),
        "total_chars": len(output),
        "est_tokens": round(len(output) / 4),
        "results": [{"title": r["title"], "url": r["url"], "chars": len(r["content"])} for r in top],
    }


async def main():
    # Pre-warm model
    print("Pre-warming model...")
    _get_model()

    results = []
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(4.0),
        headers={"User-Agent": "search-mcp/0.1 (content extraction for LLMs)"},
    ) as client:
        for qi, query in enumerate(QUERIES, 1):
            print(f"\n[{qi}/{len(QUERIES)}] {query}")
            r = await run_query(client, query)
            results.append(r)
            print(f"  {r['total_ms']}ms | {r['num_results']} results | {r['total_chars']}ch (~{r['est_tokens']}tok)")
            print(f"  Search: {r['search_ms']}ms | Fetch+Chunk: {r['extract_ms']}ms")

    # Save
    out_path = Path(__file__).parent / "baseline_our_tool.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n" + "=" * 70)
    avg_ms = sum(r["total_ms"] for r in results) / len(results)
    avg_tok = sum(r["est_tokens"] for r in results) / len(results)
    print(f"Average: {avg_ms:.0f}ms latency, ~{avg_tok:.0f} tokens/query")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
