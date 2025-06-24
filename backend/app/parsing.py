"""Utility to convert transcript files to plain text.
Currently strips numbers and timestamp lines typical in VTT/SRT files.
"""
from __future__ import annotations
import re


def parse_transcript(raw: str) -> str:
    """Return transcript text without timing artifacts.

    Parameters
    ----------
    raw:
        Raw contents of the transcript file as a single string.

    The function removes typical cues found in ``.vtt`` or ``.srt`` files:
    - Numeric index lines (e.g. ``1`` ``2`` ``3``)
    - Timestamp ranges (e.g. ``00:00:01.000 --> 00:00:05.000``)

    The cleaned lines are joined using newlines.
    """
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        # Skip sequential indices such as "1", "2", etc.
        if re.match(r"^\d+$", stripped):
            continue
        # Skip VTT/SRT timestamp lines containing "-->".
        if "-->" in stripped:
            continue
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)
