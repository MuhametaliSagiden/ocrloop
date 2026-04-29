"""Tests for the compact-layout post-processor."""

from __future__ import annotations

from ocrloop.ocr import _compact_layout


def test_question_line_is_not_bulletised():
    assert _compact_layout("1) Что такое X?") == "1) Что такое X?"


def test_indented_question_line_is_not_bulletised():
    # Even when the question itself sits at modest indent, the regex catches it.
    assert _compact_layout("   6) Что такое VPN?") == "6) Что такое VPN?"


def test_indented_answer_gets_bullet_prefix():
    text = "1) Q?\n      Option A\n      Option B"
    out = _compact_layout(text)
    assert out == "1) Q?\nО Option A\nО Option B"


def test_blank_lines_are_dropped():
    text = "1) Q?\n\n      A\n\n\n      B"
    out = _compact_layout(text)
    assert out == "1) Q?\nО A\nО B"


def test_question_continuation_keeps_no_bullet():
    # Wrapped continuation line shares the question's small indent and must
    # NOT be flagged as a bullet item.
    text = "2) Какая серия международных стандартов ISO посвящена\n  информационной безопасности?\n      ISO 9001"
    out = _compact_layout(text)
    assert out == (
        "2) Какая серия международных стандартов ISO посвящена\n"
        "информационной безопасности?\n"
        "О ISO 9001"
    )


def test_empty_input():
    assert _compact_layout("") == ""
    assert _compact_layout("\n\n") == ""


def test_idempotent_on_already_compact_text():
    text = "1) Q?\nО A\nО B"
    assert _compact_layout(text) == text


def test_extract_text_uses_compact_by_default(monkeypatch):
    """``OCR_LAYOUT`` defaults to compact: indented input gets bullets."""
    monkeypatch.delenv("OCR_LAYOUT", raising=False)
    monkeypatch.delenv("OCR_ENGINE", raising=False)

    def fake_tesseract(image_bytes: bytes, cfg) -> str:
        return "1) Q?\n\n      A\n      B"

    monkeypatch.setattr("ocrloop.ocr._tesseract_recognize", fake_tesseract)
    from ocrloop.ocr import extract_text

    assert extract_text(b"\x00") == "1) Q?\nО A\nО B"


def test_extract_text_preserve_layout_keeps_indentation(monkeypatch):
    monkeypatch.setenv("OCR_LAYOUT", "preserve")
    monkeypatch.delenv("OCR_ENGINE", raising=False)

    def fake_tesseract(image_bytes: bytes, cfg) -> str:
        return "1) Q?\n\n      A\n      B"

    monkeypatch.setattr("ocrloop.ocr._tesseract_recognize", fake_tesseract)
    from ocrloop.ocr import extract_text

    out = extract_text(b"\x00")
    # Blank line and indentation survive when explicitly opted in.
    assert "      A" in out
    assert "\n\n" in out
