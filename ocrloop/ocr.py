"""OCR pipeline.

Tesseract is used because, unlike most ML-based OCR engines, it can output
character-level coordinates (TSV) which lets us reconstruct the original layout
— line breaks, indentation and tabs — instead of receiving a flat string with
all whitespace collapsed.

The pipeline:
    1. Decode bytes to a numpy image.
    2. Light pre-processing (grayscale, upscale very small images, gentle
       binarisation) to reduce artifacts without aggressive denoising that
       could damage glyph shapes.
    3. Run Tesseract with ``--psm 6`` (assume a single uniform block of text)
       and ``preserve_interword_spaces=1`` so multi-space gaps are kept.
    4. Re-flow the recognised words from the TSV output back onto a character
       grid, restoring left-side indentation and inter-word spacing.
    5. Strip decorative bullet/bubble characters that Tesseract often invents
       (or misreads as letters), replacing them with spaces so column
       alignment is preserved.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image

# Characters that are almost always decorative in the kind of screenshots users
# OCR (chat clients, slide decks, list screenshots, …). They are replaced with
# a single space so the surrounding indentation/columns stay aligned.
_DECORATIVE_CHARS = (
    # geometric bullets
    "\u2022\u2023\u2043\u204c\u204d\u2219\u25a0\u25a1\u25aa\u25ab"
    "\u25b2\u25b3\u25b6\u25b7\u25c0\u25c1\u25c6\u25c7\u25cb\u25cf"
    "\u25d8\u25d9\u25e6"
    # stars / asterisks decorations
    "\u2605\u2606\u2729\u272a\u272b\u272c\u272d\u272e\u272f"
    # check / cross marks
    "\u2713\u2714\u2717\u2718"
    # arrows often used as bullets
    "\u2192\u2794\u279c\u279d\u279e\u279f\u27a1\u27a4\u27a8\u27ab"
    # heavy/dingbat squares
    "\u2756\u2759\u275a"
    # middle dot variants
    "\u00b7\u30fb"
)

# Numbered "bubbles": ①..⑳, ❶..❿, ⓵..⓾, etc. We collapse them to spaces too —
# they act as list numbering, not as content. The user explicitly asked for
# this so Tesseract does not "hallucinate" letters from them.
_BUBBLE_RANGES = [
    (0x2460, 0x2473),  # ① — ⑳
    (0x2474, 0x2487),  # ⑴ — ⒇
    (0x2488, 0x249b),  # ⒈ — ⒛
    (0x24eb, 0x24ff),  # ⓫ — ⓿
    (0x2776, 0x2793),  # ❶..❿, ➀..➉, ➊..➓
]

_BUBBLE_CHARS = "".join(
    chr(c) for lo, hi in _BUBBLE_RANGES for c in range(lo, hi + 1)
)

_DECORATIVE_RE = re.compile(f"[{re.escape(_DECORATIVE_CHARS + _BUBBLE_CHARS)}]")


@dataclass(frozen=True)
class OCRConfig:
    langs: str = "rus+eng"
    psm: int = 6  # assume a single uniform block of text
    oem: int = 1  # LSTM only (best quality for non-Latin)
    min_height: int = 1000  # upscale shorter images for better recognition

    def tesseract_config(self) -> str:
        return (
            f"--oem {self.oem} --psm {self.psm} "
            "-c preserve_interword_spaces=1"
        )


def _decode_image(data: bytes) -> np.ndarray:
    """Decode arbitrary image bytes into a BGR numpy array."""
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    arr = np.array(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def _preprocess(img: np.ndarray, cfg: OCRConfig) -> np.ndarray:
    """Light pre-processing: grayscale + upscale small images.

    We deliberately avoid heavy thresholding/denoising — Tesseract's LSTM model
    handles antialiased text well, and aggressive binarisation tends to thin
    Cyrillic strokes and create artifacts.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h = gray.shape[0]
    if h < cfg.min_height:
        scale = cfg.min_height / h
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    return gray


def _reflow_from_tsv(tsv: str, cfg: OCRConfig) -> str:
    """Reconstruct line breaks + indentation from Tesseract's TSV output.

    Tesseract gives us bounding boxes for every recognised word. We project
    those words onto a character grid using the median character width as the
    column unit, which preserves leading indentation and inter-word gaps far
    better than ``image_to_string`` (which collapses runs of whitespace).
    """
    rows = []
    for line in tsv.splitlines()[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        try:
            level = int(parts[0])
            page, block, par, line_no = (int(parts[1]), int(parts[2]),
                                         int(parts[3]), int(parts[4]))
            left = int(parts[6])
            width = int(parts[8])
            conf = float(parts[10])
        except ValueError:
            continue
        text = parts[11]
        if level != 5 or not text.strip() or conf < 0:
            continue
        rows.append((page, block, par, line_no, left, width, text))

    if not rows:
        return ""

    # Estimate the median character width across the document so we can convert
    # pixel offsets into character columns.
    widths = [w / max(len(t), 1) for *_, w, t in rows if t.strip()]
    char_w = max(float(np.median(widths)), 1.0) if widths else 1.0

    # Group words by (block, paragraph, line) preserving order.
    out_lines: list[str] = []
    current_key: tuple[int, int, int, int] | None = None
    buffer: list[tuple[int, str]] = []  # (column, word)

    def flush() -> None:
        if not buffer:
            out_lines.append("")
            return
        buffer.sort(key=lambda x: x[0])
        line = ""
        for col, word in buffer:
            if len(line) < col:
                line += " " * (col - len(line))
            # If a previous word already overlaps this column, just append a
            # single space — this happens when the median char width is off.
            if line and not line.endswith(" "):
                line += " "
            line += word
        out_lines.append(line.rstrip())

    last_par_key: tuple[int, int, int] | None = None
    for page, block, par, line_no, left, _w, text in rows:
        key = (page, block, par, line_no)
        par_key = (page, block, par)
        if current_key is not None and key != current_key:
            flush()
            buffer = []
            # Blank line between paragraphs/blocks for readability.
            if last_par_key is not None and par_key != last_par_key:
                out_lines.append("")
        current_key = key
        last_par_key = par_key
        col = int(round(left / char_w))
        buffer.append((col, text))
    flush()

    # Trim trailing empties.
    while out_lines and not out_lines[-1].strip():
        out_lines.pop()
    return "\n".join(out_lines)


def _is_alphanumeric_token(tok: str) -> bool:
    """True if ``tok`` contains at least one letter or digit (any script)."""
    return any(ch.isalnum() for ch in tok)


def _strip_decorations(text: str) -> str:
    """Remove decorative bullet / bubble characters while keeping layout.

    Two passes:

    1. Replace any character in the explicit decorative set with a space.
    2. If a line starts with ``<indent><short non-alphanumeric token><spaces>``
       (e.g. ``  ¢ Setup …`` after Tesseract mis-recognised ``•`` as ``¢``),
       blank that token out. We only do this when the token is short (≤2
       chars) and contains no letters or digits in any script — this matches
       Tesseract's typical bullet artefacts (``°``, ``¢``, ``@``, ``*``, …)
       without touching real content like quotes immediately followed by a
       word (``"hello"``) which has no space after them.
    """
    cleaned_lines = []
    bullet_token_re = re.compile(r"^(\s*)([^\w\s]{1,2})(\s+)(\S.*)?$", re.UNICODE)
    for line in text.splitlines():
        # Pass 1: explicit decorative chars → space.
        new = _DECORATIVE_RE.sub(" ", line)
        # Pass 2: leading short non-alphanumeric token followed by space.
        m = bullet_token_re.match(new)
        if m and not _is_alphanumeric_token(m.group(2)):
            indent, bullet, gap, rest = m.group(1), m.group(2), m.group(3), m.group(4) or ""
            new = indent + " " * len(bullet) + gap + rest
        cleaned_lines.append(new.rstrip())
    return "\n".join(cleaned_lines)


def extract_text(image_bytes: bytes, cfg: OCRConfig | None = None) -> str:
    """Run the full OCR pipeline on a single image and return clean text."""
    cfg = cfg or OCRConfig()
    img = _decode_image(image_bytes)
    processed = _preprocess(img, cfg)
    tsv = pytesseract.image_to_data(
        processed,
        lang=cfg.langs,
        config=cfg.tesseract_config(),
    )
    text = _reflow_from_tsv(tsv, cfg)
    text = _strip_decorations(text)
    return text.strip("\n")
