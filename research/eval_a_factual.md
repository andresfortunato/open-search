# Eval A: Factual Coverage Scoring

5 queries, 5 ground-truth facts each. Score: does the tool's output contain this fact?
✓ = present, ✗ = absent

## Query 1: "rust async await best practices"

| Fact | WebSearch | open-search-mcp |
|------|-----------|-----------------|
| 1. Futures are lazy (must be .await'd to run) | ✓ | ✓ |
| 2. Avoid blocking ops (thread::sleep, sync I/O) in async | ✓ mentions thread::sleep | ✓ mentions "blocking operations disrupt cooperative scheduling" |
| 3. Don't hold non-async locks across .await (deadlock risk) | ✓ explicitly | ✓ mentions async mutex vs sync mutex |
| 4. Use Send-safe types (no Rc/RefCell with multithreaded executor) | ✓ names Rc, RefCell, Send | ✗ |
| 5. Use join! for concurrent futures instead of sequential awaits | ✓ mentions join! | ✗ |

**Score: WebSearch 5/5, open-search-mcp 3/5**

## Query 2: "CRISPR gene editing mechanism explained"

| Fact | WebSearch | open-search-mcp |
|------|-----------|-----------------|
| 1. Three steps: recognition, cleavage, repair | ✓ detailed | ✓ snippet mentions it |
| 2. Guide RNA directs Cas9 to target sequence | ✓ | ✗ |
| 3. Cas9 cuts double-stranded DNA | ✓ "molecular scissors" | ✗ |
| 4. PAM site (protospacer adjacent motif) required | ✓ "3 base pair upstream to PAM" | ✗ |
| 5. Repair via NHEJ or HDR pathways | ✓ names both | ✗ |

**Score: WebSearch 5/5, open-search-mcp 1/5**

## Query 3: "what causes lithium battery thermal runaway"

| Fact | WebSearch | open-search-mcp |
|------|-----------|-----------------|
| 1. Self-sustaining exothermic chain reaction | ✓ | ✓ "uncontrollably released" |
| 2. Causes: internal short circuit, overcharging, physical damage, external heat | ✓ lists all 4 | ✓ "mechanical damage, external heat, short circuit, overcharging" |
| 3. Separator breakdown leads to electrode contact | ✓ "breakdown of internal separators" | ✗ |
| 4. Temperature thresholds (80°C SEI, 130°C separator, 150°C cathode) | ✗ | ✗ |
| 5. Positive feedback loop (heat → reaction → more heat) | ✓ describes the cycle | ✗ |

**Score: WebSearch 4/5, open-search-mcp 2/5**

## Query 4: "PostgreSQL window functions examples"

| Fact | WebSearch | open-search-mcp |
|------|-----------|-----------------|
| 1. Window functions compute across related rows without collapsing them | ✓ | ✓ |
| 2. OVER clause with PARTITION BY and ORDER BY | ✓ shows syntax | ✓ references syntax |
| 3. Common functions: ROW_NUMBER, RANK, DENSE_RANK | ✓ names RANK, DENSE_RANK | ✓ "Ranking Functions" |
| 4. LAG/LEAD for accessing previous/next rows | ✓ describes both | ✗ |
| 5. Concrete SQL example (SELECT ... OVER (PARTITION BY ...)) | ✓ shows empsalary example | ✗ |

**Score: WebSearch 5/5, open-search-mcp 3/5**

## Query 5: "how do solar panels convert light to electricity"

| Fact | WebSearch | open-search-mcp |
|------|-----------|-----------------|
| 1. Photovoltaic effect converts light to current | ✓ | ✓ |
| 2. Photons absorbed by semiconductor (silicon) | ✓ | ✓ mentions PV cell |
| 3. Electrons dislodged from atoms creating current | ✓ | ✗ |
| 4. DC output converted to AC by inverter | ✓ | ✗ |
| 5. Electric field in p-n junction pushes electrons | ✓ "electric field pushes energized electrons" | ✗ |

**Score: WebSearch 5/5, open-search-mcp 2/5**

---

## Summary

| Query | WebSearch | open-search-mcp |
|-------|-----------|-----------------|
| Rust async | 5/5 | 3/5 |
| CRISPR | 5/5 | 1/5 |
| Thermal runaway | 4/5 | 2/5 |
| PostgreSQL windows | 5/5 | 3/5 |
| Solar panels | 5/5 | 2/5 |
| **Total** | **24/25 (96%)** | **11/25 (44%)** |

## Key Finding

**WebSearch covers 2.2x more facts than open-search-mcp.** The gap is largest on science/explainer queries (CRISPR: 5 vs 1) and smallest on technical queries (Rust: 5 vs 3, PostgreSQL: 5 vs 3).

### Why our tool scores low

1. **Snippets carry almost no information.** When extraction fails and we fall back to [snippet], the result is 1-2 sentences that rarely contain specific facts. CRISPR query: 3 of 5 results were snippets.

2. **Chunk selection optimizes for relevance, not coverage.** We pick the ~500 chars most similar to the query. This discards the detailed explanation paragraphs that contain the specific facts.

3. **WebSearch's AI synthesis covers more ground.** It reads 10+ full pages and synthesizes key facts from all of them into one coherent answer. We return excerpts from 5 pages.

4. **Our 500-char budget is too tight for explainer content.** Technical queries work better because the answer is often in one focused paragraph. Science/explainer content is distributed across many paragraphs — our chunker picks one and misses the rest.
