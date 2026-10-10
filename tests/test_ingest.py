from datetime import date

import pytest

from eka.ingest import chunk_document, load_directory, load_file
from eka.models import Document

from .conftest import SAMPLE_DOCS


def test_front_matter_is_parsed():
    doc = load_file(SAMPLE_DOCS / "HR" / "leave_policy.md")
    assert doc.title == "Leave Policy"
    assert doc.department == "HR"
    assert doc.doc_type == "policy"
    assert doc.effective_date == date(2025, 4, 1)
    assert not doc.text.startswith("---")


def test_department_falls_back_to_folder(tmp_path):
    folder = tmp_path / "Legal"
    folder.mkdir()
    (folder / "nda_guide.txt").write_text("Sign the NDA before sharing drawings.", encoding="utf-8")
    doc = load_file(folder / "nda_guide.txt")
    assert doc.department == "Legal"
    assert doc.title == "Nda Guide"


def test_load_directory_reads_all_sample_docs():
    docs = load_directory(SAMPLE_DOCS)
    assert len(docs) == 8
    assert {d.department for d in docs} == {"HR", "Finance", "IT", "Operations", "Quality"}


def test_chunks_respect_size_and_carry_metadata():
    doc = load_file(SAMPLE_DOCS / "HR" / "leave_policy.md")
    chunks = chunk_document(doc, chunk_size=400, overlap=80)
    assert len(chunks) > 1
    # a chunk may exceed the target only by the overlap carried over from the previous chunk
    assert all(len(c.text) <= 400 + 80 + 2 for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert all(c.department == "HR" and c.doc_id == doc.doc_id for c in chunks)


def test_overlap_repeats_text_across_boundaries():
    text = "\n\n".join(f"Paragraph {i} talks about topic number {i} in some detail." for i in range(30))
    doc = Document(doc_id="d1", title="T", text=text, source="t.md")
    chunks = chunk_document(doc, chunk_size=200, overlap=60)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert any(word in nxt.text[:80] for word in prev.text[-60:].split())


def test_long_paragraph_is_split_on_sentences():
    long_para = " ".join(f"Sentence {i} is about the warehouse." for i in range(60))
    doc = Document(doc_id="d2", title="T", text=long_para, source="t.md")
    chunks = chunk_document(doc, chunk_size=300, overlap=0)
    assert len(chunks) > 3
    assert all(c.text.endswith(".") for c in chunks)


def test_pdf_files_are_loaded(tmp_path):
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    folder = tmp_path / "Safety"
    folder.mkdir()
    pdf = folder / "ppe_rules.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 750, "Safety shoes and helmets are mandatory on the shop floor.")
    c.showPage()
    c.drawString(72, 750, "Visitors must be escorted at all times.")
    c.save()

    doc = load_file(pdf)
    assert doc.department == "Safety"
    assert "helmets are mandatory" in doc.text and "escorted" in doc.text
