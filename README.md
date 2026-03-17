# open-search-mcp

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

Head-to-head comparison across 5 diverse queries (technical, scientific, how-to):

| Metric | open-search-mcp | WebSearch | WebSearch+WebFetch |
|--------|----------------|-----------|-------------------|
| Latency | **4.1s** | ~3s | ~6-10s (multi-call) |
| Tokens/query | **535** | ~650 | ~1,300+ |
| Tool calls | **1** | 1 | 3-6 |
| Content fidelity | **Verbatim excerpts** | AI-rewritten | AI-summarized |
| Extraction success | **76%** (90%+ with browser) | N/A | 50% |
| Auto-approvable | **Yes** | No | No |
| Code examples preserved | **Yes** | No (paraphrased) | Sometimes |

Full eval data: `research/baseline_comparison.md`

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

 One-command install:

```bash
claude mcp add open-search -- uvx --from git+https://github.com/andresfortunato/open-search.git open-search-mcp
```

That's it. Claude Code will run `uvx` to fetch and start the server on demand.

 With Playwright (recommended — improves extraction from 76% to ~100%):

```bash
# Clone for browser support (uvx doesn't support extras yet)
git clone https://github.com/andresfortunato/open-search.git
cd open-search
uv venv && uv pip install -e ".[browser]"
playwright install chromium

# Add to Claude Code
claude mcp add open-search -- $(pwd)/.venv/bin/open-search-mcp
```

### Manual config

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "open-search": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/andresfortunato/open-search.git", "open-search-mcp"]
    }
  }
}
```

### Make it the default search

Add this to your `~/.claude/CLAUDE.md` (global) or project `CLAUDE.md` so Claude Code prefers open-search over the built-in WebSearch:

```markdown
# Search
Use the `search` MCP tool (from open-search) instead of WebSearch for all web searches.
Use the `extract` MCP tool instead of WebFetch for URL content extraction.
These tools return verbatim source text with higher extraction success in a single tool call.
```

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
docker-compose.yml  # SearXNG + Redis
searxng/settings.yml  # SearXNG config
```

## License

MIT
