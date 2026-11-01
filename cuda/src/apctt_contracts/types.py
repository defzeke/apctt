"""Shared constrained types for strict contract models."""

from __future__ import annotations

import datetime as dt
import re
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints
from pydantic_core import PydanticCustomError

IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
REQUEST_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


class StrictModel(BaseModel):
    """Unknown fields rejected, no type coercion, immutable after validation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def non_blank(value: str) -> str:
    if not value.strip():
        raise PydanticCustomError(
            "non_blank_string", "value must contain a non-whitespace character"
        )
    return value


def iso_date(value: str) -> str:
    if not _DATE.fullmatch(value):
        raise PydanticCustomError("iso_date", "value must be an ISO date YYYY-MM-DD")
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        raise PydanticCustomError("iso_date", "value must be an ISO date YYYY-MM-DD") from None
    return value


def _omittable_schema(schema: dict[str, Any]) -> None:
    """Publish an omit-only field without the `null` option and without `default: null`."""
    schema.pop("default", None)
    options = schema.pop("anyOf", None)
    if options is None:
        return
    kept = [option for option in options if option != {"type": "null"}]
    if len(kept) == 1:
        schema.update(kept[0])
    else:
        schema["anyOf"] = kept


def omittable() -> Any:
    """Default for an optional `X | None` field that may be omitted but never sent as null.

    Validation is unchanged (null is rejected by `reject_null` or the result validator); only the
    published JSON Schema drops the null option.
    """
    return Field(default=None, json_schema_extra=_omittable_schema)


def reject_null(value: object) -> object:
    if value is None:
        raise PydanticCustomError("null_not_allowed", "omit the optional field instead of null")
    return value


NonBlankText = Annotated[str, AfterValidator(non_blank)]
Identifier = Annotated[str, StringConstraints(pattern=IDENTIFIER_PATTERN)]
RequestId = Annotated[str, StringConstraints(pattern=REQUEST_ID_PATTERN)]
IsoDate = Annotated[str, AfterValidator(iso_date)]
Sha256Hex = Annotated[str, StringConstraints(pattern=SHA256_PATTERN)]
