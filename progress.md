# Open Search MCP — Progress & Decisions

## Architecture (2026-03-16)

**Chunk Selection:** Option A — Top-K independent chunks, document-order output. If output reads as fragmented, consider Option B (adjacent-chunk merging) at `chunker.py:_assemble_top_k()`.

**Embedding Model:** fastembed (ONNX Runtime, ~200MB) with `all-MiniLM-L6-v2` (80MB, 384-dim). Benchmarked against BGE-small and nomic — MiniLM won decisively on speed with comparable quality.

**Playwright:** Optional dep (`[browser]`). httpx-first (76% success, ~200ms), Playwright-second for failed URLs (2.5s, recovers 403s and JS-rendered pages → ~100% extraction).

## Latency Optimization (2026-03-17)

### ONNX Padding Fix
Root cause: fastembed pads all texts in a batch to the longest text's token count. Fix: sort-by-length + batch_size=8 → 3x faster chunk selection (890ms → 286ms/page).

### Stream Processing + Early Return
`fetch_and_extract` processes pages as fetches complete via `asyncio.as_completed`. Returns early once `max_results` pages succeed, cancels remaining fetches. Playwright only triggers when needed.

### Current Latency (post all optimizations)

| Metric | **open-search-mcp** | WebSearch | WebSearch+WebFetch |
|--------|---|---|---|
| Latency | **4.1s** | ~3s | ~6-10s (multi-call) |
| Tokens/query | **535** | ~650 | ~1,300+ |
| Tool calls | **1** | 1 | 3-6 |
| Extraction success | **76-100%** | N/A | 50% |
| Auto-approvable | **Yes** | No | No |

Latency progression: 6.3s → 4.1s (35% reduction from stream processing + early return).

## Production Readiness — All Tiers Complete (2026-03-17)

### Tier 1 — Install & Docs
- [x] README with eval results, architecture, Docker docs
- [x] Install from GitHub via `uvx --from git+...` (no clone needed)
- [x] First-run UX (progress logging during model download + Docker startup)
- [x] Docker documentation (auto-start, ports, lifecycle)

### Tier 2 — Core Features
- [x] Playwright fallback (optional dep `[browser]`)
- [x] Stream processing + early return (5 tests)
- [x] URL caching with 5-min TTL (7 tests)
- [x] SearXNG error recovery with auto-restart (3 tests)

### Tier 3 — Polish
- [x] GitHub Actions CI (Python 3.11/3.12)
- [x] SearXNG secret key auto-generation
- [x] Configurable chunk budget (`OPEN_SEARCH_CHUNK_CHARS` env var)
- [x] Debug mode (`OPEN_SEARCH_DEBUG=1` logs pipeline timing)
- [x] User-Agent rotation (reduces 403s without Playwright)

### Test Suite
15 tests: extractor (7), cache (5), server (3). All passing.

## Benchmark Data
- Embedding model comparison: `research/benchmark_results.json`
- Baseline eval (vs WebSearch+WebFetch): `research/baseline_comparison.md`
- Eval outputs: `research/baseline_our_tool.json`
- Test corpus: `research/benchmark_corpus.json`
