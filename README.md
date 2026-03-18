# Open Search MCP

Open source MCP search server for Claude Code. Replaces WebSearch + WebFetch with a single tool call that returns verbatim, query-relevant excerpts.

## Why?

Claude Code's built-in search tools have significant limitations:

| Problem | WebSearch | WebFetch | open-search-mcp |
|---------|-----------|----------|----------------|
| Content type | AI-rewritten summary | AI-summarized via Haiku | Verbatim source text |
| Extraction success | N/A | 50% (fails on JS pages) | 76-90%+ |
| Tool calls per search | 3-6 (search + fetch each URL) | 1 per URL | **1 total** |
| Auto-approvable | No | No | **Yes** |
| Token cost | ~650 tok (summary only) | ~650 tok per URL | **~667 tok (5 results)** |

## Eval Results

Head-to-head comparison across 20 diverse queries (technical, scientific, practical, domain-specific):

| Metric | open-search-mcp | WebSearch | WebSearch+WebFetch |
|--------|----------------|-----------|-------------------|
| Latency | **4.5s avg** (2.7-8.1s) | ~3s | ~6-10s (multi-call) |
| Tokens/query | **~1,400** | ~700 | ~1,400+ |
| Results per query | **5** | 10 links + summary | 1-3 per WebFetch |
| Tool calls | **1** | 1 | 3-6 |
| Content fidelity | **Verbatim excerpts** | AI-rewritten | AI-summarized |
| Extraction success | **100%** (5/5 every query) | N/A | ~50% |
| Auto-approvable | **Yes** | No | No |
| Code examples preserved | **Yes** | No (paraphrased) | Sometimes |

Full eval data: `research/eval_20_results.json`

## Prerequisites

- **Python 3.10+**
- **Docker** (for SearXNG search backend)
- **uv** (recommended) or pip

### How Docker is used

open-search-mcp uses [SearXNG](https://docs.searxng.org/) as its search backend. SearXNG runs as a Docker container alongside Redis for caching.

- **Auto-start:** When the MCP server starts, it checks if SearXNG is reachable. If not, it runs `docker compose up -d` automatically. First start pulls images (~500MB) and takes ~10s. Subsequent starts take ~3s.
- **Auto-managed:** Containers run with `restart: "no"` so they don't persist after Docker restarts. The MCP server starts them on demand.
- **Port:** SearXNG listens on `localhost:8888` (configurable via `SEARXNG_URL` env var).
- **Stopping:** Run `docker compose down` in the project directory to stop containers.
- **Volume:** SearXNG settings are mounted read-only. No host files are created or modified by the containers.

## Install

```bash
# Install globally (available in all projects)
claude mcp add -s user open-search -- uvx --from git+https://github.com/andresfortunato/open-search.git open-search-mcp

# Set as default search (one-time — tells Claude to prefer open-search over WebSearch)
uvx --from git+https://github.com/andresfortunato/open-search.git open-search-mcp --setup

# Optional: install Chromium for ~100% extraction success (vs 76% without)
playwright install chromium
```

The `-s user` flag installs the MCP server globally so it's available in all projects. The `--setup` command writes a rule to `~/.claude/rules/` so Claude prefers open-search over built-in WebSearch/WebFetch everywhere. Both are one-time — they persist across sessions.

Without Chromium, the server works fine — it just can't extract from JS-rendered pages (Reddit, Medium, etc.).

## Tools

### `search`

Search the web and return extracted content. Replaces WebSearch + WebFetch in a single call.

```
search(query, max_results=5, include_domains=None, exclude_domains=None, time_range=None)
```

Returns structured results with title, URL, relevance score, and query-relevant content excerpts (~500 chars per result).

### `extract`

Extract content from specific URLs. Direct replacement for WebFetch, without the Haiku summarization.

```
extract(urls, query=None)
```

Returns clean markdown content. When `query` is provided, content is reduced to the most relevant chunks via embeddings.

## How it works

```
Query
  -> SearXNG (search 70+ engines: Google, Brave, DuckDuckGo, ...)
  -> Fetch HTML (httpx, concurrent, 4s timeout)
  -> [Playwright fallback for failed URLs, if installed]
  -> Extract content (trafilatura, F1=0.958)
  -> Split into paragraphs
  -> Embed query + paragraphs (fastembed, all-MiniLM-L6-v2, 80MB ONNX)
  -> Select top chunks by cosine similarity (~500 chars/result)
  -> BM25 rank results
  -> Return structured text
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARXNG_URL` | `http://localhost:8888` | SearXNG instance URL |
| `FETCH_TIMEOUT` | `4` | HTTP fetch timeout in seconds |
| `MAX_CONTENT_LENGTH` | `20000` | Max chars extracted per page before chunking |
| `MAX_CONCURRENT_FETCHES` | `5` | Max concurrent URL fetches |
| `OPEN_SEARCH_CHUNK_CHARS` | `500` | Target chars per result (chunk budget) |
| `OPEN_SEARCH_DEBUG` | `false` | Enable debug logging (pipeline timing per query) |
| `OPEN_SEARCH_COMPOSE_DIR` | (auto-detected) | Path to docker-compose.yml |

## Architecture

```
src/open_search_mcp/
  server.py     # MCP server, tool definitions, Docker lifecycle
  searcher.py   # SearXNG client + BM25 scoring
  extractor.py  # URL fetching + trafilatura + Playwright fallback
  chunker.py    # Embeddings-based chunk selection (fastembed)
  cache.py      # TTL-based URL cache
docker-compose.yml          # SearXNG + Redis
searxng/settings.yml.template  # SearXNG config (generated at runtime)
```

## License

MIT
