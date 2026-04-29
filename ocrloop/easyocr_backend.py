"""EasyOCR recognition backend.

EasyOCR uses CRAFT for text detection and a CRNN-based recogniser. It is
generally better than Tesseract on noisy real-world screenshots, especially
for picking the correct script in mixed Russian/English content (the most
common failure mode users see with Tesseract: a Russian word recognised as
all-Latin look-alikes, or an English word recognised as all-Cyrillic).

The trade-off is weight: EasyOCR depends on PyTorch and on first run downloads
~80 MB of model weights. We import it lazily so the module is optional —
sessions that stick with Tesseract pay no cost.

This module exposes a single ``recognize(image_bytes, cfg) -> str`` entry
point that returns text already laid out (line breaks + indentation
preserved) but **without** the decorative-symbol / Cyrillic-confusable
post-processing — those are applied uniformly downstream in ``ocr.py``.
"""

from __future__ import annotations

import io
from threading import Lock
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Map Tesseract-style language codes to EasyOCR's ISO-639-1 codes.
_TESSERACT_TO_EASYOCR: dict[str, str] = {
    "rus": "ru",
    "eng": "en",
    "ukr": "uk",
    "deu": "de",
    "fra": "fr",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "pol": "pl",
    "tur": "tr",
}

_reader_cache: dict[tuple[str, ...], Any] = {}
_reader_lock = Lock()


def _to_easyocr_langs(tesseract_langs: str) -> list[str]:
    """Translate ``rus+eng`` style codes to EasyOCR language codes."""
    parts = [p.strip() for p in tesseract_langs.split("+") if p.strip()]
    return [_TESSERACT_TO_EASYOCR.get(p, p) for p in parts]


def _get_reader(langs_str: str):
    """Lazily build (and cache) an ``easyocr.Reader`` for these languages."""
    langs = tuple(_to_easyocr_langs(langs_str))
    with _reader_lock:
        if langs not in _reader_cache:
            # Import here so the rest of the package doesn't pay the import
            # cost when the Tesseract backend is in use.
            import easyocr  # noqa: PLC0415

            _reader_cache[langs] = easyocr.Reader(
                list(langs), gpu=False, verbose=False
            )
        return _reader_cache[langs]


def _decode_image(data: bytes) -> np.ndarray:
    """Decode arbitrary image bytes into an RGB numpy array.

    EasyOCR accepts RGB (or grayscale) arrays directly, so we don't need the
    BGR conversion the Tesseract path does.
    """
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return np.array(img)


def _group_into_lines(
    regions: list[tuple[float, float, float, float, str]],
) -> list[list[tuple[float, float, float, float, str]]]:
    """Group OCR regions into lines based on vertical overlap.

    Two regions belong to the same visual line if the vertical centre of one
    falls within the y-range of the other. We extend the y-range as we add
    regions so a line whose first word is short can still grow to include
    later words on the same baseline.
    """
    if not regions:
        return []
    regions = sorted(regions, key=lambda r: (r[0], r[2]))
    lines: list[list[tuple[float, float, float, float, str]]] = []
    current = [regions[0]]
    cur_y_min, cur_y_max = regions[0][0], regions[0][1]
    for r in regions[1:]:
        y_top, y_bottom = r[0], r[1]
        centre = (y_top + y_bottom) / 2
        if cur_y_min <= centre <= cur_y_max:
            current.append(r)
            cur_y_min = min(cur_y_min, y_top)
            cur_y_max = max(cur_y_max, y_bottom)
        else:
            lines.append(current)
            current = [r]
            cur_y_min, cur_y_max = y_top, y_bottom
    lines.append(current)
    for line in lines:
        line.sort(key=lambda r: r[2])
    return lines


def _reflow_easyocr(results: list[tuple[Any, str, float]]) -> str:
    """Re-flow EasyOCR results onto a character grid, preserving layout.

    EasyOCR returns ``(bbox, text, confidence)`` per detected region, where
    ``bbox`` is a list of four ``[x, y]`` corner points (top-left, top-right,
    bottom-right, bottom-left). We:

    * Drop very low confidence regions (Tesseract's heuristic equivalent).
    * Compute the median per-character width across the document so we can
      convert pixel coordinates into character columns.
    * Group regions into visual lines by vertical overlap.
    * Project each line's regions onto a character grid, restoring leading
      indentation and inter-word gaps.
    """
    regions: list[tuple[float, float, float, float, str]] = []
    char_widths: list[float] = []
    for bbox, text, conf in results:
        if not text or conf is not None and conf < 0.1:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_left, x_right = min(xs), max(xs)
        y_top, y_bottom = min(ys), max(ys)
        if x_right <= x_left or y_bottom <= y_top:
            continue
        regions.append((y_top, y_bottom, x_left, x_right, text))
        char_widths.append((x_right - x_left) / max(len(text), 1))
    if not regions:
        return ""

    char_w = max(float(np.median(char_widths)), 1.0)
    lines = _group_into_lines(regions)

    out_lines: list[str] = []
    last_bottom: float | None = None
    for line in lines:
        if last_bottom is not None:
            gap = line[0][0] - last_bottom
            heights = [bot - top for top, bot, *_ in line]
            avg_h = float(np.median(heights)) if heights else 1.0
            if gap > avg_h * 0.8:
                out_lines.append("")
        text_line = ""
        for _y_top, _y_bot, x_left, _x_right, text in line:
            col = int(round(x_left / char_w))
            if len(text_line) < col:
                text_line += " " * (col - len(text_line))
            elif text_line and not text_line.endswith(" "):
                text_line += " "
            text_line += text
        out_lines.append(text_line.rstrip())
        last_bottom = max(r[1] for r in line)

    while out_lines and not out_lines[-1].strip():
        out_lines.pop()
    return "\n".join(out_lines)


def recognize(image_bytes: bytes, langs: str = "rus+eng") -> str:
    """Run EasyOCR on ``image_bytes`` and return reflowed (raw) text.

    No decoration stripping or confusable normalisation is applied here —
    those are handled uniformly in :func:`ocrloop.ocr.extract_text`.
    """
    reader = _get_reader(langs)
    img = _decode_image(image_bytes)
    raw = reader.readtext(img, paragraph=False)
    # EasyOCR returns confidence as float; cv2 import kept for parity with
    # the rest of the codebase even though we don't use it directly here.
    _ = cv2  # silence unused-import lint when this module is split tested
    return _reflow_easyocr(raw)
