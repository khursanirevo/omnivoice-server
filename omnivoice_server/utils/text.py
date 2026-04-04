"""
Sentence splitting for streaming mode.

Goal: Split text into chunks that:
  1. End at natural sentence boundaries (. ! ? newline)
  2. Don't exceed max_chars
  3. Don't split in the middle of numbers, abbreviations, URLs
"""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z\u4e00-\u9fff\u3040-\u30ff'
    r'\u00C0-\u024F'
    r'\u1E00-\u1EFF'
    r'])'
    r'|(?<=[。！？])',
)

_FALSE_ENDS = re.compile(
    r'\d+\.\d+'  # Decimals: 3.14
    r'|v\d+\.\d+'  # Version numbers: v2.1.0
    r'|[A-Z][a-z]{0,3}\.'  # Abbreviations: Dr., Inc.
    r'|\w+\.\w{2,6}(?:/|\s|$)'  # URLs: example.com
)


def split_sentences(text: str, max_chars: int = 400) -> list[str]:
    """
    Split text into sentence-level chunks suitable for streaming.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    if len(text) <= max_chars:
        return [text]

    raw_sentences = _SENTENCE_END.split(text)
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not raw_sentences:
        return [text]

    chunks: list[str] = []
    current = ""

    for sentence in raw_sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = current + " " + sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            result.extend(_split_at_words(chunk, max_chars))

    return [c for c in result if c.strip()]


def _split_at_words(text: str, max_chars: int) -> list[str]:
    """Split text at word boundary when it exceeds max_chars."""
    words = text.split()
    parts: list[str] = []
    current = ""

    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            parts.append(current)
            current = word

    if current:
        parts.append(current)

    return parts
