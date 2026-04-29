"""Tests for the Latin↔Cyrillic confusable normaliser."""

from __future__ import annotations

from ocrloop.confusables import normalize_confusables


def test_converts_all_latin_word_in_cyrillic_context():
    # Document-level context must be predominantly Cyrillic, so we surround
    # the test line with extra Russian text.
    src = (
        "Что такое информационная безопасность и почему это важно?\n"
        "Что Yto Takoe VPN Network?\n"
        "Это технология защиты данных в открытой сети передачи."
    )
    out = normalize_confusables(src)
    # Yto and Takoe should be rewritten because every Latin char is mappable
    # (Y→Ч, t→т, o→о; T→Т, a→а, k→к, o→о, e→е).
    assert "Что Что Такое" in out
    # VPN and Network must survive (V, N, w, r, l have no Cyrillic mapping).
    assert "VPN" in out
    assert "Network" in out


def test_leaves_english_only_lines_alone():
    src = "Quick brown fox jumps over the lazy dog"
    assert normalize_confusables(src) == src


def test_handles_mixed_script_word():
    # Tesseract reading DDoS-атака as ОВо5З-атака — fix the leading mixed
    # part. We can't restore "DDoS" but we should at least normalise the
    # Cyrillic-context Latin characters.
    src = "Что нарушает доступность системы (ОВо5З-атака)?"
    out = normalize_confusables(src)
    # Every Latin letter that's still in the word and has a mapping should
    # be converted. Non-mappable letters/digits stay.
    assert "О" in out  # was already Cyrillic О
    # The mixed token shouldn't grow new Latin letters.
    assert "В" in out


def test_preserves_layout_whitespace():
    # Indentation and inter-word spacing must be byte-identical when
    # rewriting words.
    src = (
        "Это документ преимущественно на русском языке для контекста.\n"
        "    Yto    Takoe   VPN"
    )
    out = normalize_confusables(src)
    assert out.endswith("    Что    Такое   VPN")


def test_does_not_convert_when_some_letters_unmappable():
    # "Virtual" has V, r, u, l with no Cyrillic confusable → stays English.
    src = "Что такое Virtual Private Network?"
    out = normalize_confusables(src)
    assert "Virtual" in out
    assert "Private" in out
    assert "Network" in out


def test_empty_and_blank_input():
    assert normalize_confusables("") == ""
    assert normalize_confusables("\n\n") == "\n\n"


def test_preserves_english_code_identifier_in_preserve_list():
    # ``text`` previously got rewritten to ``техт`` (t→т, e→е, x→х). Code
    # identifiers / common English UI words must survive in Cyrillic context.
    src = (
        "Атрибут отвечающий за высоту компонента или разметки:\n"
        "text\n"
        "type"
    )
    out = normalize_confusables(src)
    assert "text" in out
    assert "type" in out
    assert "техт" not in out


def test_preserves_snake_case_identifiers():
    # A line containing an underscore is treated as a code line; per-word
    # confusable conversion is skipped for that line entirely.
    src = (
        "Атрибут отвечающий за высоту компонента или разметки:\n"
        "layout_width\n"
        "layout_height"
    )
    out = normalize_confusables(src)
    assert "layout_width" in out
    assert "layout_height" in out


def test_preserves_camel_case_identifiers():
    src = (
        "Атрибут отвечающий за высоту компонента или разметки:\n"
        "onClickListener\n"
        "setAdapter"
    )
    out = normalize_confusables(src)
    assert "onClickListener" in out
    assert "setAdapter" in out
