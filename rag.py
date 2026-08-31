"""
Lightweight RAG (retrieval-augmented generation) over a curated knowledge
base of Indian land-record terminology, state-specific systems, and
due-diligence guidance.

Uses BM25 (keyword-based) retrieval rather than embeddings -- no extra API
key, no model download, no vector database. This is a deliberate choice for
a small, well-curated corpus like this one; for a much larger or more
semantically diverse corpus, dense retrieval (e.g. embeddings + a vector
store) would be the natural upgrade path.
"""

import os
import re
import glob

from rank_bm25 import BM25Okapi


KNOWLEDGE_BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "knowledge_base",
)

# How many chunks to retrieve and inject as context per query.
TOP_K = 3

# Minimum BM25 score for a chunk to be considered relevant at all.
# Prevents injecting irrelevant context on queries the knowledge base
# genuinely has nothing to do with (e.g. "what's the weather today").
MIN_SCORE = 3.0

# Common filler words filtered out before scoring, so queries like
# "what should I check" don't spuriously match on "what"/"should"/"i"
# appearing incidentally across many chunks.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
    "her", "its", "our", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "do", "does", "did", "can", "could", "should", "would", "will",
    "shall", "may", "might", "must", "to", "of", "in", "on", "at",
    "for", "with", "about", "and", "or", "but", "if", "so", "as",
    "not", "no", "yes", "please", "tell", "me", "know", "want",
}


def _tokenize(text):
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _chunk_document(text, source_name, max_words=150):
    """
    Splits a markdown document into chunks along its '## ' section
    headers, so each chunk stays topically coherent (one term/section per
    chunk) rather than splitting mid-explanation.
    """

    sections = re.split(r"\n(?=## )", text)

    chunks = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        # Very long sections get further split by word count as a
        # safety net, though our current documents don't need it.
        words = section.split()

        if len(words) <= max_words:
            chunks.append((source_name, section))

        else:
            for i in range(0, len(words), max_words):
                piece = " ".join(words[i:i + max_words])
                chunks.append((source_name, piece))

    return chunks


class LandRecordsRAG:

    def __init__(self, knowledge_base_dir=KNOWLEDGE_BASE_DIR):

        self.chunks = []  # list of (source_name, text)

        for filepath in sorted(glob.glob(os.path.join(knowledge_base_dir, "*.md"))):

            source_name = os.path.basename(filepath).replace(".md", "")

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            self.chunks.extend(_chunk_document(text, source_name))

        if not self.chunks:
            raise RuntimeError(
                f"No knowledge base documents found in {knowledge_base_dir}"
            )

        tokenized_chunks = [_tokenize(text) for _, text in self.chunks]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(self, query, top_k=TOP_K, min_score=MIN_SCORE):
        """
        Returns a list of (source_name, text, score) tuples for the
        top-k most relevant chunks, filtered to a minimum relevance score.
        """

        tokenized_query = _tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        results = [
            (source_name, text, score)
            for (source_name, text), score in ranked[:top_k]
            if score >= min_score
        ]

        return results

    def build_context_block(self, query, top_k=TOP_K, min_score=MIN_SCORE):
        """
        Convenience helper: retrieves relevant chunks and formats them
        into a single string ready to inject into a system prompt, with
        source labels for grounding/citation. Returns None if nothing
        sufficiently relevant was found.
        """

        results = self.retrieve(query, top_k=top_k, min_score=min_score)

        if not results:
            return None

        parts = []

        for source_name, text, score in results:
            parts.append(f"[Source: {source_name}]\n{text}")

        return "\n\n---\n\n".join(parts)