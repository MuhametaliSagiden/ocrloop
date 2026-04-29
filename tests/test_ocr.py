"""Unit tests for the pure helpers in ``ocrloop.ocr`` and ``ocrloop.bot``.

Tests intentionally avoid invoking Tesseract so they run anywhere without
system dependencies.
"""

from __future__ import annotations

from ocrloop.bot import _split_for_telegram
from ocrloop.ocr import _strip_decorations


def test_strip_geometric_bullets_keeps_indent():
    src = "  • Hello world\n  • Привет мир"
    expected = "    Hello world\n    Привет мир"
    assert _strip_decorations(src) == expected


def test_strip_bubble_numerals():
    src = "① First\n② Second\n❶ Bold\n⓵ Outline"
    expected = "  First\n  Second\n  Bold\n  Outline"
    assert _strip_decorations(src) == expected


def test_strip_misrecognised_bullets_via_short_token_rule():
    # Tesseract often outputs "¢", "°" or "@" instead of an actual bullet glyph.
    src = "  ¢ Setup environment\n  ° Russian\n  @ First milestone"
    out = _strip_decorations(src)
    assert "¢" not in out and "°" not in out and "@" not in out
    assert "Setup environment" in out
    assert "Russian" in out
    assert "First milestone" in out


def test_keeps_real_content_punctuation():
    # A leading quote stuck to the next word must NOT be stripped.
    assert _strip_decorations('"hello" world') == '"hello" world'
    # Numbered list markers are content, not decoration.
    assert _strip_decorations("1. step one") == "1. step one"
    assert _strip_decorations("2) step two") == "2) step two"


def test_strip_radio_button_artifacts():
    # Tesseract reads ○ / ● / ⦿ as combinations like "(О", "(О)" or a bare
    # "О" — strip them when followed by 2+ spaces, or in any bracketed form.
    src = (
        "  (О  Заудаление старых файлов\n"
        "  О   за блокировку\n"
        "  (О) Подбор пароля\n"
        "  [X] Done\n"
        "  ( ) Empty checkbox"
    )
    out = _strip_decorations(src)
    assert "(О" not in out and "(О)" not in out
    assert "[X]" not in out and "( )" not in out
    assert "Заудаление" in out
    assert "за блокировку" in out
    assert "Подбор пароля" in out
    assert "Done" in out
    assert "Empty checkbox" in out


def test_radio_rule_does_not_strip_real_o_words():
    # Russian "О компании" with a single space between "О" and the next word
    # must NOT be touched by the radio-button rule.
    assert _strip_decorations("О компании") == "О компании"
    # "OK Computer" similarly.
    assert _strip_decorations("OK Computer") == "OK Computer"


def test_strip_preserves_empty_lines():
    src = "first\n\nsecond"
    assert _strip_decorations(src) == "first\n\nsecond"


def test_split_for_telegram_under_limit():
    text = "short text"
    assert _split_for_telegram(text, limit=4096) == [text]


def test_split_for_telegram_breaks_on_newline():
    line = "x" * 100
    text = "\n".join([line] * 50)  # ~5099 chars
    chunks = _split_for_telegram(text, limit=1000)
    assert all(len(c) <= 1000 for c in chunks)
    # Recombining should give us back the same lines.
    rejoined = "\n".join(chunks)
    assert rejoined.replace("\n", "") == text.replace("\n", "")


def test_split_for_telegram_no_newline_falls_back_to_space():
    text = "word " * 500  # 2500 chars, no newline
    chunks = _split_for_telegram(text, limit=300)
    assert all(len(c) <= 300 for c in chunks)
    # Every chunk should end on a word boundary (space) except possibly last.
    for c in chunks[:-1]:
        assert c.endswith(" ") or c.split(" ")[-1] in ("", "word")
