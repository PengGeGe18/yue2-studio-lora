"""A tiny, dependency-free TOML reader for the subset this CLI's config uses.

Python 3.11 ships ``tomllib``; older interpreters use this fallback. Supported:
comments, ``[section]`` headers, ``key = value`` pairs, basic/literal strings,
integers, floats, booleans, single-line arrays and inline tables. Multi-line
strings and dotted keys are intentionally not supported.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


class TomlError(ValueError):
    pass


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def loads(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    table: Dict[str, Any] = root
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            if not name:
                raise TomlError(f"empty table name on line {lineno}")
            table = root
            for part in _split_dotted(name, lineno):
                existing = table.get(part)
                if existing is None:
                    existing = {}
                    table[part] = existing
                if not isinstance(existing, dict):
                    raise TomlError(f"{part!r} is not a table on line {lineno}")
                table = existing
            continue
        if "=" not in line:
            raise TomlError(f"expected 'key = value' on line {lineno}: {raw!r}")
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            raise TomlError(f"empty key on line {lineno}")
        parts = _split_dotted(key, lineno)
        target = table
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise TomlError(f"{part!r} is not a table on line {lineno}")
        target[parts[-1]] = _parse_value(value.strip(), lineno)
    return root


def _strip_comment(line: str) -> str:
    """Remove a ``#`` comment that is not inside a string."""
    out: List[str] = []
    quote: str | None = None
    escaped = False
    for char in line:
        if quote == '"':
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
            continue
        if quote == "'":
            out.append(char)
            if char == "'":
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
            out.append(char)
            continue
        if char == "#":
            break
        out.append(char)
    return "".join(out)


def _split_dotted(text: str, lineno: int) -> List[str]:
    parts = [part.strip() for part in text.split(".")]
    for part in parts:
        if not _BARE_KEY.match(part or ""):
            raise TomlError(f"invalid key {part!r} on line {lineno}")
    return parts


def _parse_value(text: str, lineno: int) -> Any:
    if not text:
        raise TomlError(f"missing value on line {lineno}")
    if text[0] == '"':
        return _parse_basic_string(text, lineno)
    if text[0] == "'":
        return _parse_literal_string(text, lineno)
    if text.startswith("["):
        return _parse_array(text, lineno)
    if text.startswith("{"):
        return _parse_inline_table(text, lineno)
    if text == "true":
        return True
    if text == "false":
        return False
    return _parse_number(text, lineno)


def _parse_basic_string(text: str, lineno: int) -> str:
    if len(text) < 2 or not text.endswith('"'):
        raise TomlError(f"unterminated string on line {lineno}")
    body = text[1:-1]
    escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "b": "\b", "f": "\f"}
    out: List[str] = []
    index = 0
    while index < len(body):
        char = body[index]
        if char != "\\":
            out.append(char)
            index += 1
            continue
        index += 1
        if index >= len(body):
            raise TomlError(f"dangling escape on line {lineno}")
        code = body[index]
        if code in escapes:
            out.append(escapes[code])
            index += 1
            continue
        if code in ("u", "U"):
            width = 4 if code == "u" else 8
            digits = body[index + 1:index + 1 + width]
            if len(digits) != width:
                raise TomlError(f"bad unicode escape on line {lineno}")
            out.append(chr(int(digits, 16)))
            index += 1 + width
            continue
        raise TomlError(f"unknown escape \\{code} on line {lineno}")
    return "".join(out)


def _parse_literal_string(text: str, lineno: int) -> str:
    if len(text) < 2 or not text.endswith("'"):
        raise TomlError(f"unterminated string on line {lineno}")
    return text[1:-1]


def _parse_number(text: str, lineno: int) -> Any:
    cleaned = text.replace("_", "")
    try:
        if any(char in cleaned for char in ".eE") and not cleaned.lower().startswith(("0x", "0o", "0b")):
            return float(cleaned)
        return int(cleaned, 0)
    except ValueError as error:
        raise TomlError(f"invalid value {text!r} on line {lineno}") from error


def _split_items(body: str, lineno: int) -> List[str]:
    items: List[str] = []
    depth = 0
    quote: str | None = None
    escaped = False
    current: List[str] = []
    for char in body:
        if quote:
            current.append(char)
            if quote == '"':
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quote = None
            elif char == "'":
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
            current.append(char)
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        if char == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    if quote or depth != 0:
        raise TomlError(f"unbalanced value on line {lineno}")
    return items


def _parse_array(text: str, lineno: int) -> List[Any]:
    if not text.endswith("]"):
        raise TomlError(f"unterminated array on line {lineno}")
    body = text[1:-1].strip()
    if not body:
        return []
    return [_parse_value(item, lineno) for item in _split_items(body, lineno)]


def _parse_inline_table(text: str, lineno: int) -> Dict[str, Any]:
    if not text.endswith("}"):
        raise TomlError(f"unterminated inline table on line {lineno}")
    body = text[1:-1].strip()
    table: Dict[str, Any] = {}
    if not body:
        return table
    for item in _split_items(body, lineno):
        if "=" not in item:
            raise TomlError(f"invalid inline table entry {item!r} on line {lineno}")
        key, _, value = item.partition("=")
        table[key.strip()] = _parse_value(value.strip(), lineno)
    return table
