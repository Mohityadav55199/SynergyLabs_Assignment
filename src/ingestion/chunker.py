import hashlib
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from src.config import settings

class Chunk(BaseModel):
    chunk_id: str
    chunk_hash: str
    doc_id: str
    filename: str
    format: str
    chunk_index: int
    text: str
    section: str
    metadata: Dict[str, Any]

class DocumentChunker:
    """
    Configurable recursive character text splitter with overlap.
    Preserves document structure and computes deterministic SHA-256 hashes
    to enable 100% idempotent ingestion and eliminate duplicate vectors.
    """
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        # `is None` (not `or`) so an explicit overlap of 0 is respected.
        self.chunk_size = settings.default_chunk_size if chunk_size is None else chunk_size
        self.chunk_overlap = settings.default_chunk_overlap if chunk_overlap is None else chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive (got {self.chunk_size})")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0 (got {self.chunk_overlap})")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})")

    def _overlap_tail(self, chunk: str) -> str:
        """Last ~chunk_overlap characters of `chunk`, advanced to a word boundary so overlap never starts mid-word."""
        if self.chunk_overlap == 0 or len(chunk) <= self.chunk_overlap:
            return ""
        tail = chunk[-self.chunk_overlap:]
        first_space = tail.find(" ")
        return tail[first_space + 1:] if 0 <= first_space < len(tail) - 1 else tail

    @staticmethod
    def _merge_tiny_chunks(chunks: List[str], min_chars: int = 100) -> List[str]:
        """Prepends chunks shorter than `min_chars` (typically a stranded heading) to the next chunk."""
        merged: List[str] = []
        carry = ""
        for c in chunks:
            c = f"{carry}\n{c}".strip() if carry else c
            carry = ""
            if len(c) < min_chars:
                carry = c
            else:
                merged.append(c)
        if carry:
            if merged:
                merged[-1] = f"{merged[-1]}\n{carry}"
            else:
                merged.append(carry)
        return merged

    def _split_text(self, text: str) -> List[str]:
        """Recursively splits text using the hierarchy of separators."""
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []

        # Find best separator
        chosen_sep = ""
        for sep in self.separators:
            if sep in text:
                chosen_sep = sep
                break

        joiner = chosen_sep
        if chosen_sep in (". ", "? ", "! "):
            # Split after sentence-final punctuation but keep it, so no sentence loses its full stop at a chunk edge.
            splits = re.split(r"(?<=[.?!])\s+", text)
            joiner = " "
        elif chosen_sep:
            splits = text.split(chosen_sep)
        else:
            # Fallback to direct character slicing
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        chunks = []
        current_chunk = ""

        for part in splits:
            candidate = f"{current_chunk}{joiner}{part}" if current_chunk else part
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If a single part exceeds chunk_size, recursively split it with next separators
                if len(part) > self.chunk_size:
                    sub_splitter = DocumentChunker(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separators=self.separators[self.separators.index(chosen_sep) + 1:] if chosen_sep in self.separators else [""]
                    )
                    sub_chunks = sub_splitter._split_text(part)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    # Apply overlap from the tail of current_chunk if possible
                    overlap_prefix = self._overlap_tail(current_chunk)
                    current_chunk = f"{overlap_prefix} {part}".strip()

        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_document(
        self,
        doc_id: str,
        filename: str,
        doc_format: str,
        text: str,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Splits parsed document text into typed, indexed chunks with SHA-256 fingerprinting.
        """
        text = re.sub(r"[ \t]+", " ", text)          # collapse HTML/PDF indentation runs
        raw_chunks = self._merge_tiny_chunks(self._split_text(text))
        chunks: List[Chunk] = []
        current_section = "General"

        for idx, chunk_text in enumerate(raw_chunks):
            # Check for header indicator in chunk
            header_match = re.search(r"^(?:#+\s*(.+)|((?:Section|Chapter)\s+\d+.*))$", chunk_text, re.MULTILINE | re.IGNORECASE)
            if header_match:
                current_section = (header_match.group(1) or header_match.group(2)).strip()[:80]

            # Deterministic chunk hash based on content and position
            hash_input = f"{doc_id}:{idx}:{chunk_text.strip()}"
            chunk_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
            chunk_id = f"{doc_id}_c{idx:03d}_{chunk_hash[:8]}"

            chunk_meta = dict(base_metadata or {})
            chunk_meta.update({
                "char_length": len(chunk_text),
                "chunk_index": idx,
                "section": current_section,
                "format": doc_format,
                "filename": filename
            })

            chunks.append(Chunk(
                chunk_id=chunk_id,
                chunk_hash=chunk_hash,
                doc_id=doc_id,
                filename=filename,
                format=doc_format,
                chunk_index=idx,
                text=chunk_text,
                section=current_section,
                metadata=chunk_meta
            ))

        return chunks
