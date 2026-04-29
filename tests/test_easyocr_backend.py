"""Tests for the EasyOCR backend.

We don't import easyocr in tests — it's a heavy optional dependency. We
exercise the pure layout-reflow helpers with synthetic bounding-box input
and verify the engine selector dispatches correctly via monkeypatching.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from ocrloop.easyocr_backend import (
    _group_into_lines,
    _reflow_easyocr,
    _to_easyocr_langs,
)


def test_to_easyocr_langs_translates_known_codes():
    assert _to_easyocr_langs("rus+eng") == ["ru", "en"]
    assert _to_easyocr_langs("rus") == ["ru"]
    assert _to_easyocr_langs("eng+rus+deu") == ["en", "ru", "de"]


def test_to_easyocr_langs_passes_through_unknown():
    # Unknown codes are forwarded as-is so users can pass EasyOCR-native codes.
    assert _to_easyocr_langs("ja") == ["ja"]


def test_group_into_lines_splits_on_vertical_gap():
    # Three regions: two on the same line, one below.
    regions = [
        (10.0, 30.0, 0.0, 50.0, "Hello"),
        (12.0, 32.0, 60.0, 110.0, "world"),
        (60.0, 80.0, 0.0, 50.0, "Next"),
    ]
    lines = _group_into_lines(regions)
    assert len(lines) == 2
    assert [r[4] for r in lines[0]] == ["Hello", "world"]
    assert [r[4] for r in lines[1]] == ["Next"]


def test_group_into_lines_sorts_within_line_by_x():
    # Regions arrive out of order; line sorting must put them left-to-right.
    regions = [
        (10.0, 30.0, 100.0, 150.0, "third"),
        (10.0, 30.0, 0.0, 40.0, "first"),
        (10.0, 30.0, 50.0, 90.0, "second"),
    ]
    lines = _group_into_lines(regions)
    assert len(lines) == 1
    assert [r[4] for r in lines[0]] == ["first", "second", "third"]


def _bbox(x_left: float, y_top: float, x_right: float, y_bottom: float):
    return [
        [x_left, y_top],
        [x_right, y_top],
        [x_right, y_bottom],
        [x_left, y_bottom],
    ]


def test_reflow_preserves_indentation():
    # Two lines: "Header" at column 0, "  body text" indented further.
    results: list[tuple[Any, str, float]] = [
        (_bbox(0, 0, 60, 20), "Header", 0.95),
        (_bbox(40, 30, 140, 50), "body", 0.95),
        (_bbox(150, 30, 200, 50), "text", 0.95),
    ]
    out = _reflow_easyocr(results)
    lines = out.splitlines()
    assert len(lines) == 2
    assert lines[0].lstrip() == "Header"
    # body should be more indented than Header
    assert lines[1].startswith(" ")
    assert lines[1].lstrip().startswith("body")


def test_reflow_drops_low_confidence_regions():
    results: list[tuple[Any, str, float]] = [
        (_bbox(0, 0, 60, 20), "real", 0.9),
        (_bbox(0, 30, 60, 50), "noise", 0.05),  # below 0.1 threshold
    ]
    out = _reflow_easyocr(results)
    assert "real" in out
    assert "noise" not in out


def test_reflow_handles_empty_input():
    assert _reflow_easyocr([]) == ""


def test_extract_text_dispatches_to_easyocr_via_env(monkeypatch):
    """``OCR_ENGINE=easyocr`` must route through the EasyOCR backend."""
    monkeypatch.setenv("OCR_ENGINE", "easyocr")

    # Stand in for the real EasyOCR module so importing the backend works
    # without pulling PyTorch.
    fake_called = {}

    def fake_recognize(image_bytes: bytes, langs: str) -> str:
        fake_called["langs"] = langs
        fake_called["bytes_len"] = len(image_bytes)
        return "Test"

    fake_module = type(sys)("ocrloop.easyocr_backend")
    fake_module.recognize = fake_recognize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ocrloop.easyocr_backend", fake_module)

    from ocrloop.ocr import extract_text

    out = extract_text(b"\x00\x01\x02")
    assert fake_called["langs"] == "rus+eng"
    assert fake_called["bytes_len"] == 3
    assert out == "Test"


def test_extract_text_default_is_tesseract(monkeypatch):
    """Without ``OCR_ENGINE`` set the Tesseract path runs."""
    monkeypatch.delenv("OCR_ENGINE", raising=False)
    called = {"hit": False}

    def fake_tesseract(image_bytes: bytes, cfg) -> str:
        called["hit"] = True
        return "OK"

    monkeypatch.setattr("ocrloop.ocr._tesseract_recognize", fake_tesseract)
    from ocrloop.ocr import extract_text

    assert extract_text(b"\x00") == "OK"
    assert called["hit"] is True


@pytest.mark.parametrize("engine", ["TESSERACT", "Easyocr", "  easyocr  "])
def test_engine_selector_normalises_case_and_whitespace(engine, monkeypatch):
    monkeypatch.setenv("OCR_ENGINE", engine)
    from ocrloop.ocr import _selected_engine

    assert _selected_engine() in {"tesseract", "easyocr"}
