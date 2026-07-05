"""Regex-based PE tag processing: strip existing PE=\\d+, normalise spaces, append new PE tag."""
from __future__ import annotations
import re

_PE_PATTERN = re.compile(r"\bPE=\d+")
_SPACES = re.compile(r" +")


def tag_header(header: str, pe_tag: str) -> str:
    """Remove any PE=\\d+ tag, collapse multiple spaces, and append *pe_tag* (e.g. 'PE=1')."""
    # strip existing PE tags
    header = _PE_PATTERN.sub("", header)
    # normalise spaces
    header = _SPACES.sub(" ", header).strip()
    return f"{header} {pe_tag}"