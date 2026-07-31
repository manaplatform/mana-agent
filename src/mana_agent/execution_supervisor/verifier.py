"""Completion-contract and artifact verification."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Callable

from mana_agent.execution_supervisor.models import (
    CompletionArtifact,
    CompletionContract,
    CompletionContractType,
    VerificationReport,
    VerificationStatus,
    utc_now,
)

CustomVerifier = Callable[[CompletionContract, Path, dict], tuple[bool, dict]]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactVerifier:
    def __init__(self) -> None:
        self._custom: dict[str, CustomVerifier] = {}

    def register(self, name: str, verifier: CustomVerifier) -> None:
        if not name.strip():
            raise ValueError("custom verifier name is required")
        self._custom[name] = verifier

    def verify(
        self,
        contracts: list[CompletionContract],
        *,
        workspace: Path,
        result_payload: dict,
        attempt_started_at=None,
    ) -> VerificationReport:
        root = workspace.expanduser().resolve()
        checks: list[dict] = []
        artifacts: list[CompletionArtifact] = []
        if not contracts:
            return VerificationReport(
                status=VerificationStatus.FAILED,
                checks=[{"passed": False, "reason": "no completion contract was defined"}],
            )
        for contract in contracts:
            passed, detail, artifact = self._verify_one(
                contract, root=root, result_payload=result_payload, attempt_started_at=attempt_started_at
            )
            checks.append(
                {
                    "contract_type": contract.contract_type.value,
                    "passed": passed,
                    **detail,
                }
            )
            if artifact is not None:
                artifacts.append(artifact)
        return VerificationReport(
            status=VerificationStatus.PASSED if all(item["passed"] for item in checks) else VerificationStatus.FAILED,
            checks=checks,
            artifacts=artifacts,
        )

    def _verify_one(
        self, contract: CompletionContract, *, root: Path, result_payload: dict, attempt_started_at
    ) -> tuple[bool, dict, CompletionArtifact | None]:
        kind = contract.contract_type
        if kind in {CompletionContractType.FILE_EXISTS, CompletionContractType.DIRECTORY_EXISTS}:
            candidate = (root / contract.path).resolve()
            if not _inside(candidate, root):
                return False, {"path": contract.path, "reason": "artifact is outside the allowed workspace"}, None
            expected_file = kind == CompletionContractType.FILE_EXISTS
            exists = candidate.is_file() if expected_file else candidate.is_dir()
            size = candidate.stat().st_size if exists and expected_file else None
            checksum = _sha256(candidate) if exists and expected_file else ""
            produced = None
            if exists and attempt_started_at is not None:
                produced = candidate.stat().st_mtime >= attempt_started_at.timestamp()
            artifact = CompletionArtifact(
                artifact_type="file" if expected_file else "directory",
                path=str(candidate.relative_to(root)),
                exists=exists,
                size=size,
                sha256=checksum,
                verified_at=utc_now(),
                produced_by_attempt=produced,
            )
            passed = exists
            reason = ""
            declared_kind = "file" if expected_file else "directory"
            if contract.expected_kind not in {"any", declared_kind}:
                passed, reason = False, (
                    f"completion contract type {declared_kind} conflicts with expected kind "
                    f"{contract.expected_kind}"
                )
            elif not exists:
                reason = "claimed artifact does not exist or has the wrong type"
            elif expected_file and size is not None and size < contract.minimum_size:
                passed, reason = False, "artifact is smaller than the completion contract minimum"
            elif contract.expected_sha256 and checksum != contract.expected_sha256:
                passed, reason = False, "artifact checksum does not match"
            elif contract.require_attempt_change and produced is not True:
                passed, reason = False, "artifact was not produced or modified by this attempt"
            return passed, {"path": contract.path, "reason": reason}, artifact
        if kind == CompletionContractType.STRUCTURED_RESULT_VALID:
            required = [str(item) for item in contract.metadata.get("required_keys", [])]
            missing = [key for key in required if key not in result_payload]
            expected = dict(contract.metadata.get("expected_values", {}))
            mismatched = {
                key: {"expected": value, "actual": result_payload.get(key)}
                for key, value in expected.items()
                if result_payload.get(key) != value
            }
            return not missing and not mismatched, {
                "missing_keys": missing,
                "mismatched_values": mismatched,
            }, None
        if kind == CompletionContractType.COMMAND_SUCCEEDED:
            exit_code = result_payload.get("exit_code")
            return exit_code == 0, {"exit_code": exit_code}, None
        if kind == CompletionContractType.REMOTE_RESOURCE_CONFIRMED:
            confirmation = result_payload.get("remote_confirmation")
            return bool(confirmation), {"confirmed": bool(confirmation)}, None
        if kind == CompletionContractType.GIT_DIFF_PRESENT:
            argv = ["git", "diff", "--name-only", "HEAD"]
            if contract.path:
                argv.extend(["--", contract.path])
            completed = subprocess.run(
                argv, cwd=root, capture_output=True, text=True, check=False, shell=False
            )
            changed = [line for line in completed.stdout.splitlines() if line.strip()]
            passed = completed.returncode == 0 and bool(changed)
            artifact = CompletionArtifact(
                artifact_type="git_diff",
                path=contract.path,
                exists=passed,
                verified_at=utc_now(),
                details={"changed_files": changed},
            )
            return passed, {
                "git_returncode": completed.returncode,
                "changed_files": changed,
            }, artifact
        if kind == CompletionContractType.GIT_COMMIT_EXISTS:
            commit = contract.commit or str(result_payload.get("commit") or "")
            if not commit:
                return False, {"reason": "no commit was supplied"}, None
            completed = subprocess.run(
                ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=root,
                capture_output=True,
                check=False,
                shell=False,
            )
            return completed.returncode == 0, {"commit": commit}, None
        if kind == CompletionContractType.CUSTOM_VERIFIER:
            verifier = self._custom.get(contract.verifier_name)
            if verifier is None:
                return False, {"reason": "custom verifier is not registered"}, None
            passed, detail = verifier(contract, root, result_payload)
            json.dumps(detail, default=str)
            return bool(passed), dict(detail), None
        return False, {"reason": f"unsupported completion contract: {kind.value}"}, None
