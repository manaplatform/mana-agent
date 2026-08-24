from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, Field

from mana_agent.utils.tool_results import json_safe_dumps, json_safe_tool_payload


class SampleEnum(str, Enum):
    ALPHA = "alpha"
    BETA = "beta"


class SampleModel(BaseModel):
    name: str
    tags: set[str] = Field(default_factory=set)
    sub_tags: frozenset[str] = Field(default_factory=frozenset)
    enum_val: SampleEnum = SampleEnum.ALPHA


@dataclass
class SampleDataclass:
    id: str
    categories: set[str]


def test_json_safe_tool_payload_converts_sets_and_frozensets_to_lists() -> None:
    payload = {
        "set_strings": {"apple", "banana", "cherry"},
        "frozenset_strings": frozenset({"x", "y"}),
        "empty_set": set(),
        "nested_set": {"outer": [{"inner_set": {1, 2, 3}}]},
    }
    result = json_safe_tool_payload(payload)

    assert result["set_strings"] == ["apple", "banana", "cherry"]
    assert result["frozenset_strings"] == ["x", "y"]
    assert result["empty_set"] == []
    assert result["nested_set"]["outer"][0]["inner_set"] == [1, 2, 3]

    # Verify standard json.dumps succeeds without default handler
    serialized = json.dumps(result)
    assert "apple" in serialized
    assert "banana" in serialized


def test_json_safe_tool_payload_preserves_existing_json_types() -> None:
    payload = {
        "str": "hello",
        "int": 42,
        "float": 3.14,
        "bool_true": True,
        "bool_false": False,
        "none": None,
        "list": [1, "two", 3.0, None],
        "dict": {"a": 1, "b": "c"},
    }
    result = json_safe_tool_payload(payload)

    assert result == payload
    assert isinstance(result["bool_true"], bool)
    assert isinstance(result["int"], int)
    assert isinstance(result["float"], float)


def test_json_safe_tool_payload_handles_pydantic_models_with_sets() -> None:
    model = SampleModel(name="test_item", tags={"t1", "t2"}, sub_tags=frozenset({"s1"}))
    result = json_safe_tool_payload(model)

    assert result["name"] == "test_item"
    assert result["tags"] == ["t1", "t2"]
    assert result["sub_tags"] == ["s1"]
    assert result["enum_val"] == "alpha"

    # Must be valid for json.dumps
    json.dumps(result)


def test_json_safe_tool_payload_handles_dataclasses_with_sets() -> None:
    dc = SampleDataclass(id="dc-1", categories={"cat1", "cat2"})
    result = json_safe_tool_payload(dc)

    assert result["id"] == "dc-1"
    assert result["categories"] == ["cat1", "cat2"]
    json.dumps(result)


def test_json_safe_tool_payload_handles_datetimes_uuids_paths_decimals() -> None:
    now = datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc)
    uid = uuid4()
    p = Path("/tmp/test.txt")
    dec_int = Decimal("10.00")
    dec_float = Decimal("10.50")

    payload = {
        "time": now,
        "uuid": uid,
        "path": p,
        "dec_int": dec_int,
        "dec_float": dec_float,
    }
    result = json_safe_tool_payload(payload)

    assert result["time"] == "2026-08-25T12:00:00+00:00"
    assert result["uuid"] == str(uid)
    assert result["path"] == str(p)
    assert result["dec_int"] == 10
    assert result["dec_float"] == 10.5
    json.dumps(result)


def test_json_safe_dumps_serializes_cleanly() -> None:
    data = {
        "labels": {"INBOX", "UNREAD"},
        "nested": {"tags": frozenset({"starred"})},
    }
    dumped = json_safe_dumps(data)
    loaded = json.loads(dumped)

    assert loaded["labels"] == ["INBOX", "UNREAD"]
    assert loaded["nested"]["tags"] == ["starred"]
