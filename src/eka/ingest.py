"""Loading files into Documents and splitting them into overlapping Chunks.

Supported inputs: .md, .txt and .pdf. Markdown files may start with a simple
front-matter block to carry metadata used for filtering:

    ---
    title: Leave Policy
    department: HR
    doc_type: policy
    effective_date: 2025-04-01
    ---
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path

from .models import Chunk, Document

SUPPORTED = {".md", ".txt", ".pdf"}
_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    return meta, text[match.end():]


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_file(path: str | Path) -> Document:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {path.name}")

    raw = _read_pdf(path) if suffix == ".pdf" else path.read_text(encoding="utf-8")
    meta, body = _parse_front_matter(raw) if suffix == ".md" else ({}, raw)

    # Department falls back to the parent folder name (e.g. docs/HR/leave.pdf -> HR).
    department = meta.get("department") or (path.parent.name if path.parent.name not in {"", "."} else None)
    title = meta.get("title") or path.stem.replace("_", " ").replace("-", " ").title()
    doc_id = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]

    return Document(
        doc_id=doc_id,
        title=title,
        text=body.strip(),
        source=path.name,
        department=department,
        doc_type=meta.get("doc_type"),
        effective_date=_parse_date(meta.get("effective_date")),
    )


def load_directory(root: str | Path) -> list[Document]:
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    return [load_file(p) for p in files]


def _split_paragraphs(text: str) -> list[str]:
    # Keep markdown headings attached to the paragraph that follows them.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    merged: list[str] = []
    for block in blocks:
        if merged and merged[-1].startswith("#") and "\n" not in merged[-1]:
            merged[-1] = f"{merged[-1]}\n{block}"
        else:
            merged.append(block)
    return merged


def _overlap_tail(text: str, overlap: int) -> str:
    """The last whole sentences of `text` that fit in `overlap` characters.

    Starting the overlap on a sentence boundary avoids chunks that open mid-sentence
    ("...and the finance controller."), which read badly when quoted as evidence.
    """
    if overlap <= 0:
        return ""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n", "\n".join(lines)) if s.strip()]
    tail: list[str] = []
    for sentence in reversed(sentences):
        if sum(len(t) + 1 for t in tail) + len(sentence) > overlap:
            break
        tail.insert(0, sentence.strip())
    return " ".join(tail)


def chunk_document(doc: Document, chunk_size: int = 700, overlap: int = 120) -> list[Chunk]:
    """Paragraph-aware chunking.

    Paragraphs are packed into chunks of up to ``chunk_size`` characters. Paragraphs
    longer than that are split on sentence boundaries. Each new chunk starts with
    the last ``overlap`` characters of the previous one so facts spanning a boundary
    are still retrievable.
    """
    pieces: list[str] = []
    for para in _split_paragraphs(doc.text):
        if len(para) <= chunk_size:
            pieces.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for sentence in sentences:
            if buf and len(buf) + len(sentence) + 1 > chunk_size:
                pieces.append(buf)
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
        if buf:
            pieces.append(buf)

    chunks_text: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > chunk_size:
            chunks_text.append(current)
            tail = _overlap_tail(current, overlap)
            current = f"{tail}\n\n{piece}".strip() if tail else piece
        else:
            current = f"{current}\n\n{piece}".strip()
    if current:
        chunks_text.append(current)

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}-{i:03d}",
            doc_id=doc.doc_id,
            title=doc.title,
            text=text,
            source=doc.source,
            position=i,
            department=doc.department,
            doc_type=doc.doc_type,
            effective_date=doc.effective_date,
        )
        for i, text in enumerate(chunks_text)
    ]
