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
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .confusables import normalize_confusables

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
    # Target minimum of (height, width) after upscaling. Most quiz / Google
    # Forms screenshots users send are 700–1100 px wide; bumping them to
    # ~1600 px consistently recovers small identifier text like
    # ``layout_width`` that EasyOCR otherwise mangles.
    min_dim: int = 1600

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


def _upscale_if_small(img: np.ndarray, min_dim: int) -> np.ndarray:
    """Upscale ``img`` so the smaller of (height, width) reaches ``min_dim``.

    Uses bicubic interpolation — empirically better than Lanczos on text of
    the 12–14 px stroke height typical of browser screenshots, and faster.
    A no-op when the image is already at or above the target size.
    """
    if img.size == 0:
        return img
    h, w = img.shape[:2]
    smallest = min(h, w)
    if smallest >= min_dim:
        return img
    scale = min_dim / smallest
    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


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
    return _upscale_if_small(gray, cfg.min_dim)


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


# Characters Tesseract typically mis-recognises empty / filled radio-button
# and checkbox glyphs (○, ●, ⦿, ◉, ☐, ☑) as. When we see a short token built
# only from these, hugging the start of a line and followed by 2+ spaces, it
# is overwhelmingly an artifact rather than real content.
_RADIO_CHARS = "OoОо0()[]{}|/\\-—–"

_RADIO_RE = re.compile(
    r"^(\s*)([" + re.escape(_RADIO_CHARS) + r"]{1,3})(\s{2,})(\S.*)$",
    re.UNICODE,
)

# Bracketed markers like ``(O)``, ``(О)``, ``[X]``, ``[ ]``, ``(•)``, ``(©)``:
# a single inner character (or space) wrapped in matching parens/brackets. We
# strip these with just one trailing space because the bracket pair is itself
# a strong signal — there is no real Russian/English content shaped like
# ``(letter)<space><word>`` at the start of a line.
#
# The inner-character set covers letters Tesseract substitutes for ○/●/⦿
# (``O``, ``o``, ``О``, ``о``, ``0``, ``x``, ``X``), the bullet glyphs
# themselves, and copyright/registered/section signs Tesseract reports when
# the radio-button glyph is filled (``©``, ``®``, ``§``).
_BRACKETED_RE = re.compile(
    r"^(\s*)([(\[][OoОо0xX•●○◯◉©®§ ][)\]])(\s+)(\S.*)$",
    re.UNICODE,
)


def _strip_decorations(text: str) -> str:
    """Remove decorative bullet / bubble characters while keeping layout.

    Three passes:

    1. Replace any character in the explicit decorative set with a space.
    2. If a line starts with ``<indent><short non-alphanumeric token><spaces>``
       (e.g. ``  ¢ Setup …`` after Tesseract mis-recognised ``•`` as ``¢``),
       blank that token out. We only do this when the token is short (≤2
       chars) and contains no letters or digits in any script — this matches
       Tesseract's typical bullet artefacts (``°``, ``¢``, ``@``, ``*``, …)
       without touching real content like quotes immediately followed by a
       word (``"hello"``) which has no space after them.
    3. Strip *radio-button artefacts*: short tokens at the start of a line
       built only from ``O``/``о``/``0``/parentheses/brackets and followed
       by **two or more** spaces. Tesseract reads ○/●/⦿ as ``(О``, ``(О)``
       or a lone ``О`` and our pass-2 rule misses them because Cyrillic
       ``О`` is a letter. The 2-space requirement avoids stripping real
       content like ``О компании`` (one space) or ``OK Computer``.
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
        # Pass 3: radio-button mis-recognitions (bare token + 2+ spaces).
        m = _RADIO_RE.match(new)
        if m:
            indent, token, gap, rest = m.group(1), m.group(2), m.group(3), m.group(4)
            new = indent + " " * len(token) + gap + rest
        # Pass 4: bracketed markers like ``(O)``, ``[X]`` (single space ok).
        m = _BRACKETED_RE.match(new)
        if m:
            indent, token, gap, rest = m.group(1), m.group(2), m.group(3), m.group(4)
            new = indent + " " * len(token) + gap + rest
        cleaned_lines.append(new.rstrip())
    return "\n".join(cleaned_lines)


# Bullet glyph used to mark list / answer-option lines in compact-layout
# output. Cyrillic ``О`` is what the source screenshots use (radio-button
# circle that Tesseract / EasyOCR may or may not recognise) so we adopt it
# verbatim — it composes naturally with surrounding Russian content and
# matches the look of the competing OCR bot the user benchmarked against.
_COMPACT_BULLET = "О "

# Lines starting with ``<digits>)`` (with optional leading whitespace) are
# treated as question / list headings — never bulletised, even when they
# share the indentation of an answer option that just happens to wrap.
_QUESTION_RE = re.compile(r"^\s*\d+\)\s")

# A line indented this many columns or more (and not a question heading) is
# treated as a bullet item. The threshold is calibrated for the source
# screenshots users send: questions sit at 0–3 column indent, answers at
# 5–10. Wrapped continuation lines of a question typically share the
# question's small indent (2–3), so 4 is the sweet spot.
_BULLET_INDENT_THRESHOLD = 4

# EasyOCR sometimes recognises the empty / filled circle next to each option
# as a lone ``О``/``O`` followed by whitespace. When the line is already
# flagged as a bullet by the indent rule, we must strip that leading glyph
# before prepending our own ``О ``, otherwise the output doubles up as
# ``О О <option>``.
_LEADING_BULLET_GLYPH = re.compile(r"^[ОO]\s+")


def _compact_layout(text: str) -> str:
    """Flatten indentation and restore ``О`` markers on bullet-style lines.

    Two simple invariants:

    1. **Bullet detection.** A non-question line indented at least
       :data:`_BULLET_INDENT_THRESHOLD` columns is an answer option and is
       prefixed with ``О ``. Question headings (``1) …``) are recognised
       by regex and never bulletised, regardless of indent.
    2. **Compaction.** Leading whitespace is dropped from every line and
       blank lines are removed so the output looks like the competing
       bot's: one question per line, then ``О <option>`` rows directly
       beneath.

    The function is idempotent on already-compact text (no indentation →
    nothing flagged → output equals input minus blank lines).
    """
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue  # drop blank lines entirely
        if _QUESTION_RE.match(line):
            out.append(stripped)
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent >= _BULLET_INDENT_THRESHOLD:
            body = _LEADING_BULLET_GLYPH.sub("", stripped)
            out.append(_COMPACT_BULLET + body)
        else:
            out.append(stripped)
    return "\n".join(out)


def _selected_layout() -> str:
    """Resolve the layout style from ``OCR_LAYOUT`` (defaults to compact)."""
    return os.environ.get("OCR_LAYOUT", "compact").strip().lower()


def _tesseract_recognize(image_bytes: bytes, cfg: OCRConfig) -> str:
    """Tesseract recognition + bounding-box reflow (no post-processing)."""
    img = _decode_image(image_bytes)
    processed = _preprocess(img, cfg)
    tsv = pytesseract.image_to_data(
        processed,
        lang=cfg.langs,
        config=cfg.tesseract_config(),
    )
    return _reflow_from_tsv(tsv, cfg)


def _selected_engine() -> str:
    """Resolve the OCR engine name from ``OCR_ENGINE`` (defaults to tesseract)."""
    return os.environ.get("OCR_ENGINE", "tesseract").strip().lower()


def extract_text(image_bytes: bytes, cfg: OCRConfig | None = None) -> str:
    """Run the full OCR pipeline on a single image and return clean text.

    The recognition engine is selected by the ``OCR_ENGINE`` environment
    variable: ``tesseract`` (default) or ``easyocr``. Decoration stripping
    and Cyrillic-confusable normalisation are applied uniformly to both
    backends so output formatting is identical.
    """
    cfg = cfg or OCRConfig()
    engine = _selected_engine()
    if engine == "easyocr":
        # Lazy import — keeps the EasyOCR / PyTorch dependency optional.
        from .easyocr_backend import recognize as easyocr_recognize  # noqa: PLC0415

        text = easyocr_recognize(image_bytes, langs=cfg.langs)
    else:
        text = _tesseract_recognize(image_bytes, cfg)
    text = _strip_decorations(text)
    text = normalize_confusables(text)
    if _selected_layout() == "compact":
        text = _compact_layout(text)
    return text.strip("\n")
