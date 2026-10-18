"""Ground truth for retrieval evaluation.

A chunk is RELEVANT to a question iff it comes from the question's target file AND its text contains the
question's `evidence` string (a verbatim, answer-bearing phrase picked by hand from the source document).
The chunk set is rebuilt from the corpus with the same parser/chunker/settings as ingestion, so chunk ids
match the ids stored in the vector DB. Relevance is therefore defined over the whole corpus, which makes
Recall@k a real recall (denominator = all relevant chunks), not an assumed one.
"""
import re
from pathlib import Path
from typing import Dict, List, Set

from src.config import settings
from src.ingestion.chunker import Chunk, DocumentChunker
from src.ingestion.parsers import DocumentParser

SUPPORTED = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


def build_corpus_chunks(corpus_dir: Path, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[Chunk] = []
    for f in sorted(Path(corpus_dir).rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED:
            doc = DocumentParser.parse_file(f)
            chunks += chunker.chunk_document(doc.doc_id, doc.filename, doc.format, doc.text, doc.metadata)
    return chunks


def relevant_chunk_ids(case: Dict, chunks: List[Chunk]) -> Set[str]:
    if not case.get("is_answerable", True) or not case.get("evidence"):
        return set()
    needle = _norm(case["evidence"])
    return {
        c.chunk_id for c in chunks
        if c.filename == case["target_file"] and needle in _norm(c.text)
    }
