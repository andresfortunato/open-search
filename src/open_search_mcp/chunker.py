"""Embeddings-based chunk selection: split content into paragraphs,
embed query + paragraphs, return the most relevant chunks."""

import logging
import os

import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# Lazy-initialized global model (loaded once, reused across requests)
_model: TextEmbedding | None = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk sizing (TARGET_TOTAL_CHARS configurable via env var)
MIN_CHUNK_CHARS = 50
MAX_CHUNK_CHARS = 1000
TARGET_TOTAL_CHARS = int(os.environ.get("OPEN_SEARCH_CHUNK_CHARS", "1500"))


def _get_model() -> TextEmbedding:
    """Load the embedding model on first use."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def _split_paragraphs(text: str) -> list[str]:
    """Split markdown text into paragraph-sized chunks.

    Splits on double-newline boundaries. Merges very short chunks with
    neighbors; splits very long chunks at sentence boundaries.
    """
    raw = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    buffer = ""

    for para in raw:
        if len(para) > MAX_CHUNK_CHARS:
            # Flush buffer first
            if buffer:
                chunks.append(buffer)
                buffer = ""
            # Split long paragraph at sentence boundaries
            sentences = para.replace(". ", ".\n").split("\n")
            current = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 > MAX_CHUNK_CHARS and current:
                    chunks.append(current.strip())
                    current = sent
                else:
                    current = f"{current} {sent}".strip() if current else sent
            if current:
                chunks.append(current.strip())
        elif len(buffer) + len(para) + 2 < MIN_CHUNK_CHARS * 2:
            # Merge short paragraphs
            buffer = f"{buffer}\n\n{para}".strip() if buffer else para
        else:
            if buffer:
                chunks.append(buffer)
            buffer = para

    if buffer:
        chunks.append(buffer)

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector a and matrix b."""
    # a: (dim,), b: (n, dim) → returns (n,)
    norm_a = np.linalg.norm(a)
    norms_b = np.linalg.norm(b, axis=1)
    # Avoid division by zero
    denom = norm_a * norms_b
    denom = np.where(denom == 0, 1e-10, denom)
    return np.dot(b, a) / denom


def select_chunks(
    query: str,
    content: str,
    target_chars: int = TARGET_TOTAL_CHARS,
) -> str:
    """Select the most query-relevant chunks from content.

    Splits content into paragraphs, embeds them alongside the query,
    and returns top chunks by cosine similarity up to target_chars.

    TODO: Implement the assembly strategy here.
    Currently uses top-K independent selection.
    """
    chunks = _split_paragraphs(content)
    if not chunks:
        return content[:target_chars]

    # If content is already short enough, return as-is
    total = sum(len(c) for c in chunks)
    if total <= target_chars:
        return content

    model = _get_model()

    # Sort chunks by length before batching to minimize ONNX padding waste.
    # fastembed pads all texts in a sub-batch to the longest — sorting keeps
    # sub-batches homogeneous, reducing wasted compute by 2-3x.
    indexed_chunks = sorted(enumerate(chunks), key=lambda ic: len(ic[1]))
    sorted_chunks = [c for _, c in indexed_chunks]
    original_indices = [i for i, _ in indexed_chunks]

    all_texts = [query] + sorted_chunks
    # Small batch_size ensures sorted texts form homogeneous sub-batches,
    # so short chunks aren't padded to the length of the longest chunk.
    embeddings = list(model.embed(all_texts, batch_size=8))
    emb_array = np.array(embeddings)

    query_emb = emb_array[0]
    sorted_embs = emb_array[1:]

    # Unsort embeddings back to original chunk order
    chunk_embs = np.empty_like(sorted_embs)
    for new_pos, orig_idx in enumerate(original_indices):
        chunk_embs[orig_idx] = sorted_embs[new_pos]

    # Score each chunk against the query
    scores = _cosine_similarity(query_emb, chunk_embs)

    # TODO: This is where the assembly strategy lives.
    # Currently: pick top chunks by score until we hit the char budget.
    # Alternative: merge adjacent high-scoring chunks for coherence.
    return _assemble_top_k(chunks, scores, target_chars)


def _assemble_top_k(
    chunks: list[str],
    scores: np.ndarray,
    target_chars: int,
) -> str:
    """Assemble output by picking highest-scoring chunks up to budget.

    Selected chunks are returned in their original document order
    so the output reads coherently.
    """
    ranked_indices = np.argsort(scores)[::-1]

    selected_indices: list[int] = []
    char_count = 0
    for idx in ranked_indices:
        chunk_len = len(chunks[idx])
        if char_count + chunk_len > target_chars and selected_indices:
            break
        selected_indices.append(int(idx))
        char_count += chunk_len

    # Preserve original document order
    selected_indices.sort()
    return "\n\n".join(chunks[i] for i in selected_indices)
