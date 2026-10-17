import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import pypdf
from bs4 import BeautifulSoup

class ParsedDocument(BaseModel):
    doc_id: str
    filename: str
    format: str
    text: str
    metadata: Dict[str, Any]

def compute_file_hash(file_path: Path) -> str:
    """Computes SHA-256 hash of a file for change detection and idempotency."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

class DocumentParser:
    @staticmethod
    def parse_pdf(file_path: Path) -> ParsedDocument:
        reader = pypdf.PdfReader(str(file_path))
        text_parts = []
        page_info = []

        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
                page_info.append({"page": idx + 1, "char_count": len(page_text)})

        full_text = "\n\n".join(text_parts)
        file_hash = compute_file_hash(file_path)

        return ParsedDocument(
            doc_id=f"pdf_{file_hash[:12]}",
            filename=file_path.name,
            format="pdf",
            text=full_text,
            metadata={
                "file_hash": file_hash,
                "total_pages": len(reader.pages),
                "page_info": page_info,
                "doc_type": "technical_report"
            }
        )

    @staticmethod
    def parse_html(file_path: Path) -> ParsedDocument:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style", "noscript"]):
            script_or_style.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else file_path.stem

        # Extract text while preserving structure
        lines = []
        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote"]):
            text = element.get_text(separator=" ", strip=True)
            if text:
                if element.name.startswith("h"):
                    lines.append(f"\n## {text}\n")
                else:
                    lines.append(text)

        full_text = "\n".join(lines) if lines else soup.get_text(separator="\n", strip=True)
        file_hash = compute_file_hash(file_path)

        return ParsedDocument(
            doc_id=f"html_{file_hash[:12]}",
            filename=file_path.name,
            format="html",
            text=full_text,
            metadata={
                "title": title,
                "file_hash": file_hash,
                "doc_type": "developer_guide"
            }
        )

    @staticmethod
    def parse_markdown(file_path: Path) -> ParsedDocument:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        file_hash = compute_file_hash(file_path)
        
        # Extract title from first H1 if present
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else file_path.stem

        return ParsedDocument(
            doc_id=f"md_{file_hash[:12]}",
            filename=file_path.name,
            format="md",
            text=content,
            metadata={
                "title": title,
                "file_hash": file_hash,
                "doc_type": "architecture_spec"
            }
        )

    @classmethod
    def parse_file(cls, file_path: Path) -> ParsedDocument:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return cls.parse_pdf(file_path)
        elif suffix in [".html", ".htm"]:
            return cls.parse_html(file_path)
        elif suffix in [".md", ".markdown", ".txt"]:
            return cls.parse_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported formats: PDF, HTML, MD.")
