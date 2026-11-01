"""Strict JSON for untrusted HTTP bodies and worker outputs (SPEC-05, SPEC-09).

Duplicate keys, NaN/Infinity (including overflowing literals), invalid UTF-8, a byte order mark
and lone surrogates are rejected with a fixed reason code. The reason never contains input text.
"""

from __future__ import annotations

import json
import math
from typing import TypeAlias

# Not a PEP 695 `type` statement: participant modules may import these contracts in a worker
# environment older than 3.12, where that syntax does not parse.
JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]  # noqa: UP040

REASONS = frozenset(
    {
        "byte_order_mark",
        "invalid_utf8",
        "duplicate_key",
        "non_finite_number",
        "number_out_of_range",
        "syntax",
        "too_deep",
        "lone_surrogate",
    }
)


class StrictJSONError(ValueError):
    def __init__(self, reason: str, *, line: int | None = None, column: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.line = line
        self.column = column


class _DuplicateKey(Exception):
    pass


class _NonFinite(Exception):
    pass


def _object(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey
        result[key] = value
    return result


def _constant(_: str) -> float:
    raise _NonFinite


def _float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _NonFinite
    return value


def _reject_surrogates(value: JSONValue) -> None:
    stack: list[JSONValue] = [value]
    while stack:
        item = stack.pop()
        texts: list[str] = []
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, dict):
            texts.extend(item.keys())
            stack.extend(item.values())
        for text in texts:
            try:
                text.encode("utf-8")
            except UnicodeEncodeError:
                raise StrictJSONError("lone_surrogate") from None


def loads_strict(data: bytes) -> JSONValue:
    if data.startswith(b"\xef\xbb\xbf"):
        raise StrictJSONError("byte_order_mark")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise StrictJSONError("invalid_utf8") from None
    try:
        value: JSONValue = json.loads(
            text, object_pairs_hook=_object, parse_constant=_constant, parse_float=_float
        )
    except _DuplicateKey:
        raise StrictJSONError("duplicate_key") from None
    except _NonFinite:
        raise StrictJSONError("non_finite_number") from None
    except json.JSONDecodeError as exc:
        raise StrictJSONError("syntax", line=exc.lineno, column=exc.colno) from None
    except RecursionError:
        raise StrictJSONError("too_deep") from None
    except ValueError:
        # int() digit limit for very long integer literals.
        raise StrictJSONError("number_out_of_range") from None
    _reject_surrogates(value)
    return value
