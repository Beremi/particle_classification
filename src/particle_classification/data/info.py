from __future__ import annotations

import re
from pathlib import Path
from typing import Any


INFO_HEADER_RE = re.compile(r'^"(?P<key>[^"]+)"(?:\s+\("(?P<description>[^"]*)"\))?:$')


def parse_info_file(path: str | Path) -> dict[str, Any]:
    return parse_info_text(Path(path).read_text(encoding="utf-8", errors="replace"))


def parse_info_text(text: str) -> dict[str, Any]:
    """Parse Pixet `.t3pa.info` sidecar files into a flat metadata mapping."""

    lines = [line.strip() for line in text.splitlines()]
    out: dict[str, Any] = {}
    i = 0
    while i < len(lines):
        match = INFO_HEADER_RE.match(lines[i])
        if not match:
            i += 1
            continue

        key = _slug(match.group("key"))
        description = match.group("description") or ""
        type_spec = lines[i + 1].strip() if i + 1 < len(lines) else ""
        value_line = ""
        j = i + 2
        while j < len(lines):
            value_line = lines[j].strip()
            j += 1
            if value_line:
                break

        out[key] = _parse_value(type_spec, value_line)
        if description:
            out[f"{key}_description"] = description
        out[f"{key}_type"] = type_spec
        i = j

    return out


def _parse_value(type_spec: str, value_line: str) -> Any:
    if not value_line:
        return ""
    base_type = type_spec.split("[", 1)[0].strip().lower()
    if base_type == "char":
        return value_line

    parts = value_line.split()
    if base_type in {"double", "float"}:
        values: list[Any] = [float(part) for part in parts]
    elif base_type.startswith(("u", "i")):
        values = [int(part) for part in parts]
    else:
        values = parts

    return values[0] if len(values) == 1 else values


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"
