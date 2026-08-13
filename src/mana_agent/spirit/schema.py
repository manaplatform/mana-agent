"""Versioned Spirit schema.

Spirit may define only Mana identity, root temperament, and the relationship
between that identity and a runtime model. It is not a policy, memory, skill,
security, or coding layer.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


SPIRIT_SCHEMA_KIND = "spirit"
DEFAULT_SPIRIT_ID = "mana"
DEFAULT_SPIRIT_VERSION = 1


class TemperamentTrait(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    meaning: str = Field(min_length=1)

    @field_validator("meaning")
    @classmethod
    def normalize_meaning(cls, value: str) -> str:
        text = " ".join(str(value or "").split())
        if not text:
            raise ValueError("temperament meaning must be non-empty")
        return text


class SpiritIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    product: str = Field(min_length=1)

    @field_validator("name", "product")
    @classmethod
    def normalize_identity_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("spirit identity fields must be non-empty")
        return text


class SpiritTemperament(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    curious: TemperamentTrait
    bold: TemperamentTrait
    calm: TemperamentTrait


class SpiritRef(BaseModel):
    """Durable Spirit identifier. Checkpoints must persist this, not prompt text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    version: int = Field(ge=1)

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            raise ValueError("spirit id must be non-empty")
        return text

    def key(self) -> str:
        return f"{self.id}/{self.version}"


class Spirit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    version: int = Field(ge=1)
    identity: SpiritIdentity
    temperament: SpiritTemperament

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            raise ValueError("spirit id must be non-empty")
        return text

    def ref(self) -> SpiritRef:
        return SpiritRef(id=self.id, version=self.version)


class SpiritSettings(BaseModel):
    """Operator-selectable Spirit reference. Temperament is not configurable."""

    model_config = ConfigDict(extra="forbid")

    id: str = DEFAULT_SPIRIT_ID
    version: int = Field(default=DEFAULT_SPIRIT_VERSION, ge=1)

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        text = str(value or "").strip().lower()
        return text or DEFAULT_SPIRIT_ID

    def ref(self) -> SpiritRef:
        return SpiritRef(id=self.id, version=self.version)
