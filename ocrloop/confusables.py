"""Latin → Cyrillic confusable normalisation.

Tesseract's ``rus+eng`` mode picks a script per word. When a Cyrillic word's
glyphs happen to look just like Latin ones (and many do — Cyrillic А, В, С, Е,
Н, К, М, О, Р, Т, Х all share visual forms with Latin letters) Tesseract may
output the whole word in Latin script. The result is contamination that's
visually indistinguishable from the original (``Takoe`` vs ``Такое``) but
breaks search, copy-paste, and looks wrong.

We post-process: in lines whose surrounding context is predominantly Cyrillic,
re-encode all-Latin words back to Cyrillic — but **only** when every Latin
letter in the word has a known Cyrillic counterpart. This conservatively
preserves real English words like ``VPN``, ``USA`` or ``Virtual`` (they
contain unmappable letters like ``V``/``S``/``U``/``r``/``l``) and only
rewrites the words that Tesseract clearly miscategorised.
"""

from __future__ import annotations

import re
from typing import Final

# Visual look-alikes (forward map: Latin → Cyrillic).
# Only ASCII keys with a single Cyrillic codepoint as value.
LATIN_TO_CYRILLIC: Final[dict[str, str]] = {
    # Strong visual look-alikes
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "I": "І",
    "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х",
    "a": "а", "c": "с", "e": "е", "i": "і", "o": "о", "p": "р",
    "x": "х", "y": "у",
    # Glyphs Tesseract substitutes frequently in Cyrillic words even though
    # they aren't strict visual confusables: Y/Ч (top of Ч resembles Y in
    # some sans-serif fonts), t/т (italic т is tall and looks Latin), k/к,
    # m/м (italic Russian м is rendered like Latin m).
    "Y": "Ч", "k": "к", "t": "т", "m": "м",
}

_CYRILLIC_RANGE = re.compile(r"[\u0400-\u04FF]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"\S+")

# Lines containing these signals are almost certainly source code / UI
# attribute names, even if every letter in every token is technically a
# Latin↔Cyrillic confusable. We pass such lines through unchanged so that
# identifiers like ``layout_width`` or ``camelCase`` survive verbatim.
_CODE_LINE_HINTS = re.compile(r"[_<>{}\[\]/=]|[a-z][A-Z]|[A-Za-z]\d|\d[A-Za-z]")

# Short English words composed entirely of mappable letters (and therefore
# wrongly rewritten by the naive confusable pass). These are common enough
# in UI text and code snippets that the false-positive risk dominates the
# Cyrillic-rescue benefit. We keep the list intentionally tiny — every
# addition trades a Russian-word rescue for an English-word preservation.
_ENGLISH_PRESERVE: Final[frozenset[str]] = frozenset(
    {
        "at",
        "it",
        "me",
        "my",
        "no",
        "ok",
        "on",
        "to",
        "id",  # defensive: d isn't in the map today but behaviour shouldn't change
        "cat",
        "eat",
        "hat",
        "key",
        "map",
        "set",
        "tea",
        "test",
        "text",
        "time",
        "type",
        "case",
        "date",
        "name",
        "root",
        "save",
        "site",
        "host",
        "port",
        "path",
        "true",
        "tree",
    }
)


def _document_is_cyrillic(text: str) -> bool:
    """True if the *whole document* is predominantly Cyrillic.

    Tesseract sometimes writes an entire line in Latin script even when the
    surrounding document is clearly Russian (it picks the language per word
    and short lines may not contain a single correctly-recognised Cyrillic
    char). Deciding the language at the document level rather than per line
    lets us still rewrite those lines.
    """
    cyr = len(_CYRILLIC_RANGE.findall(text))
    lat = len(_LATIN_LETTER.findall(text))
    return cyr > lat


def _normalize_word(word: str) -> str:
    """Convert an all-Latin word to Cyrillic if every letter is mappable.

    Non-letter characters (digits, punctuation, the ``-`` in ``DDoS-атака``)
    are left untouched. Words that contain even a single un-mappable Latin
    letter (e.g. ``V``, ``S``, ``r``, ``l``) are returned unchanged so real
    English borrowings like ``VPN`` survive.
    """
    if not _LATIN_LETTER.search(word):
        return word  # nothing Latin to convert
    if _CYRILLIC_RANGE.search(word):
        # Mixed-script word: convert Latin letters that are mappable, leave
        # the rest alone. This handles cases like ``ОВо5З`` where Tesseract
        # mixed scripts within one word.
        return "".join(LATIN_TO_CYRILLIC.get(c, c) for c in word)
    # Strip surrounding punctuation when checking the preserve list so that
    # "(text)" and "text," still match "text".
    stripped = re.sub(r"^\W+|\W+$", "", word)
    if stripped.lower() in _ENGLISH_PRESERVE:
        return word
    # All-Latin word: only convert if **every** letter is mappable.
    if all((not _LATIN_LETTER.match(c)) or c in LATIN_TO_CYRILLIC for c in word):
        return "".join(LATIN_TO_CYRILLIC.get(c, c) for c in word)
    return word


def normalize_confusables(text: str) -> str:
    """Apply Latin→Cyrillic confusable correction line by line.

    Only lines whose surrounding context is predominantly Cyrillic are
    rewritten — pure-English text passes through unchanged.
    """
    if not _document_is_cyrillic(text):
        return text

    out_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        body, sep = line, ""
        if line.endswith("\n"):
            body, sep = line[:-1], "\n"
        # Skip lines that look like code / UI attribute listings — they tend
        # to contain identifiers like ``layout_width`` or ``camelCase`` that
        # the per-word pass would otherwise corrupt.
        if _CODE_LINE_HINTS.search(body):
            out_lines.append(body + sep)
            continue
        # Walk word by word, preserving the original whitespace runs so that
        # indentation/columns stay byte-aligned.
        last = 0
        rebuilt: list[str] = []
        for m in _WORD_RE.finditer(body):
            rebuilt.append(body[last : m.start()])  # whitespace block
            rebuilt.append(_normalize_word(m.group()))
            last = m.end()
        rebuilt.append(body[last:])
        out_lines.append("".join(rebuilt) + sep)
    return "".join(out_lines)
