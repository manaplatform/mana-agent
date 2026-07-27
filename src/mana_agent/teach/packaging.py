"""Validated, deterministic and non-executable .mana-flow packages."""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .models import ManaFlow, TeachError
from .redaction import Redactor


PACKAGE_FILES = {
    "manifest.yaml",
    "flow.yaml",
    "permissions.yaml",
    "selectors.json",
    "verification.yaml",
    "README.md",
}
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class ManaFlowPackager:
    def __init__(self, redactor: Redactor | None = None):
        self.redactor = redactor or Redactor()

    def export(self, flow: ManaFlow, destination: str | Path) -> Path:
        payload = flow.model_dump(mode="json", by_alias=True, exclude_none=True)
        findings = self.redactor.scan(payload)
        if findings:
            raise TeachError(
                "Flow export blocked because unsafe content remains: "
                + ", ".join(findings)
                + ". Replace or remove it before export."
            )
        files = self._files(flow, payload)
        checksum = _package_checksum({key: value for key, value in files.items() if key != "manifest.yaml"})
        manifest = {
            "package_schema_version": 1,
            "flow_id": flow.id,
            "flow_version": flow.version,
            "name": flow.name,
            "description": flow.description,
            "supported_platforms": flow.supported_platforms,
            "required_applications": flow.required_applications,
            "required_capabilities": flow.required_capabilities,
            "required_mana_agent_version": ">=0.1.1",
            "input_schema": {key: value.model_dump(mode="json") for key, value in flow.inputs.items()},
            "permission_summary": flow.permissions,
            "package_checksum": checksum,
            "creation_timestamp": flow.created_at.isoformat(),
            "redaction_report": {"findings": [], "safe": True},
            "activation": "dry_run_required",
        }
        files["manifest.yaml"] = yaml.safe_dump(manifest, sort_keys=True, allow_unicode=True).encode()
        target = Path(destination).expanduser().resolve()
        if target.suffix != ".mana-flow":
            target = target.with_suffix(".mana-flow")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(files):
                info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                archive.writestr(info, files[name])
        return target

    def import_package(self, package: str | Path) -> ManaFlow:
        source = Path(package).expanduser().resolve()
        try:
            with zipfile.ZipFile(source) as archive:
                names = set(archive.namelist())
                if names != PACKAGE_FILES:
                    raise TeachError("Invalid .mana-flow structure.")
                for info in archive.infolist():
                    path = PurePosixPath(info.filename)
                    if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(info.external_attr >> 16):
                        raise TeachError("Unsafe path or link in .mana-flow package.")
                    if info.file_size > 5_000_000:
                        raise TeachError(".mana-flow member exceeds the size limit.")
                files = {name: archive.read(name) for name in names}
        except (OSError, zipfile.BadZipFile) as exc:
            raise TeachError(f"Invalid .mana-flow package: {exc}") from exc
        manifest = yaml.safe_load(files["manifest.yaml"])
        if not isinstance(manifest, dict) or manifest.get("package_schema_version") != 1:
            raise TeachError("Unsupported .mana-flow manifest.")
        checksum = _package_checksum({key: value for key, value in files.items() if key != "manifest.yaml"})
        if checksum != manifest.get("package_checksum"):
            raise TeachError(".mana-flow checksum validation failed.")
        flow_payload = yaml.safe_load(files["flow.yaml"])
        findings = self.redactor.scan(flow_payload)
        if findings:
            raise TeachError("Imported flow contains unsafe content: " + ", ".join(findings))
        flow = ManaFlow.model_validate(flow_payload)
        flow.status = "imported_pending"
        return flow

    @staticmethod
    def _files(flow: ManaFlow, payload: dict[str, Any]) -> dict[str, bytes]:
        selectors = {
            step.id: [candidate.model_dump(mode="json") for candidate in step.selectors]
            for step in flow.steps
            if step.selectors
        }
        return {
            "flow.yaml": yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).encode(),
            "permissions.yaml": yaml.safe_dump({"permissions": flow.permissions}, sort_keys=True).encode(),
            "selectors.json": (json.dumps(selectors, indent=2, sort_keys=True) + "\n").encode(),
            "verification.yaml": yaml.safe_dump(
                {"verify": [rule.model_dump(mode="json") for rule in flow.verify]}, sort_keys=False
            ).encode(),
            "README.md": (
                f"# {flow.name}\n\n{flow.description}\n\n"
                "Imported flows are untrusted until reviewed, input-mapped, dry-run, and explicitly activated.\n"
            ).encode(),
            "manifest.yaml": b"",
        }


def _package_checksum(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(files[name])
    return digest.hexdigest()
