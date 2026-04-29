"""Tests for the compact-layout post-processor."""

from __future__ import annotations

import numpy as np

from ocrloop.ocr import OCRConfig, _compact_layout, _upscale_if_small


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


def test_leading_bullet_glyph_is_not_duplicated():
    # EasyOCR read the radio-button glyph as a lone ``О`` before the option
    # text; compact layout must not double it into ``О О option``.
    text = "1) Q?\n      О Option A\n      O Option B"
    out = _compact_layout(text)
    assert out == "1) Q?\nО Option A\nО Option B"


def test_upscale_if_small_pads_tiny_image():
    tiny = np.zeros((100, 200, 3), dtype=np.uint8)
    out = _upscale_if_small(tiny, min_dim=600)
    # Upscaling is anchored on the smallest dimension reaching the target.
    assert out.shape[0] >= 600
    assert out.shape[1] >= 600


def test_upscale_if_small_no_op_when_already_large():
    big = np.zeros((2000, 3000, 3), dtype=np.uint8)
    out = _upscale_if_small(big, min_dim=1600)
    assert out.shape == big.shape


def test_ocr_config_default_min_dim_is_reasonable():
    # Protect against accidental regression of the default.
    assert OCRConfig().min_dim >= 1200


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
