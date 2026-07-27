from __future__ import annotations

import json
import io
import zipfile
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class GitHubApiError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"GitHub API request failed ({status}): {message[:500]}")


class GitHubClient:
    def __init__(self, api_url: str = "https://api.github.com") -> None:
        self.api_url = api_url.rstrip("/")

    def request(self, method: str, path: str, *, token: str, payload: dict[str, Any] | None = None) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.api_url}{path}", data=data, method=method,
            headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "mana-agent-github-app"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                message = str(json.loads(body).get("message") or "request failed")
            except (ValueError, AttributeError):
                message = "request failed"
            raise GitHubApiError(exc.code, message) from None
        except urllib.error.URLError as exc:
            raise GitHubApiError(0, f"transport error: {exc.reason}") from None

    def request_bytes(self, path: str, *, token: str, maximum_bytes: int = 5_000_000) -> bytes:
        request = urllib.request.Request(f"{self.api_url}{path}", headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "mana-agent-github-app"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read(maximum_bytes + 1)[:maximum_bytes]
        except urllib.error.HTTPError as exc:
            raise GitHubApiError(exc.code, "binary evidence request failed") from None
        except urllib.error.URLError as exc:
            raise GitHubApiError(0, f"transport error: {exc.reason}") from None

    def create_installation_token(self, installation_id: int, app_jwt: str, repository_id: int) -> dict[str, Any]:
        return self.request("POST", f"/app/installations/{installation_id}/access_tokens", token=app_jwt, payload={"repository_ids": [repository_id]})

    def permission(self, repository: str, actor: str, token: str) -> str:
        data = self.request("GET", f"/repos/{repository}/collaborators/{urllib.parse.quote(actor, safe='')}/permission", token=token)
        return str(data.get("permission") or "none").lower()

    def team_membership(self, organization: str, team_slug: str, actor: str, token: str) -> bool:
        data = self.request("GET", f"/orgs/{urllib.parse.quote(organization)}/teams/{urllib.parse.quote(team_slug)}/memberships/{urllib.parse.quote(actor)}", token=token)
        return str(data.get("state") or "").lower() == "active"

    def repository(self, repository: str, token: str) -> dict[str, Any]:
        return self.request("GET", f"/repos/{repository}", token=token)

    def create_or_update_pr(self, repository: str, token: str, *, number: int | None, title: str, head: str, base: str, body: str, draft: bool) -> dict[str, Any]:
        if number:
            return self.request("PATCH", f"/repos/{repository}/pulls/{number}", token=token, payload={"title": title, "body": body})
        return self.request("POST", f"/repos/{repository}/pulls", token=token, payload={"title": title, "head": head, "base": base, "body": body, "draft": draft})

    def find_open_pull_request(self, repository: str, token: str, branch: str, session_id: str = "") -> dict[str, Any] | None:
        owner = repository.partition("/")[0]
        head = urllib.parse.quote(f"{owner}:{branch}")
        rows = self.request("GET", f"/repos/{repository}/pulls?state=open&head={head}&per_page=10", token=token)
        if isinstance(rows, list) and rows:
            return dict(rows[0])
        if session_id:
            candidates = self.request("GET", f"/repos/{repository}/pulls?state=open&per_page=100", token=token)
            marker = f"<!-- mana-github-task:{session_id} -->"
            return next((dict(item) for item in candidates if marker in str(item.get("body") or "")), None)
        return None

    def comment(self, repository: str, issue_number: int, token: str, body: str) -> dict[str, Any]:
        return self.request("POST", f"/repos/{repository}/issues/{issue_number}/comments", token=token, payload={"body": body})

    def workflow_evidence(self, repository: str, run_id: int, token: str) -> dict[str, Any]:
        jobs = self.request("GET", f"/repos/{repository}/actions/runs/{run_id}/jobs?filter=latest&per_page=100", token=token)
        run = self.request("GET", f"/repos/{repository}/actions/runs/{run_id}", token=token)
        sha = urllib.parse.quote(str(run.get("head_sha") or ""))
        commits = self.request("GET", f"/repos/{repository}/commits?sha={sha}&per_page=10", token=token) if sha else []
        annotations: list[dict[str, Any]] = []
        for job in jobs.get("jobs", []):
            check_url = str(job.get("check_run_url") or "")
            if check_url.startswith(self.api_url):
                rows = self.request("GET", check_url.removeprefix(self.api_url) + "/annotations?per_page=100", token=token)
                if isinstance(rows, list):
                    annotations.extend(rows)
        logs: dict[str, str] = {}
        try:
            archive = self.request_bytes(f"/repos/{repository}/actions/runs/{run_id}/logs", token=token)
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                for name in bundle.namelist()[:20]:
                    if name.endswith("/"):
                        continue
                    with bundle.open(name) as handle:
                        logs[name] = handle.read(100_000).decode("utf-8", errors="replace")
        except (GitHubApiError, zipfile.BadZipFile):
            logs = {}
        return {"jobs": jobs.get("jobs", []), "run": run, "annotations": annotations, "logs": logs, "recent_commits": commits}

    def review_evidence(self, repository: str, pull_number: int, token: str) -> dict[str, Any]:
        reviews = self.request("GET", f"/repos/{repository}/pulls/{pull_number}/reviews?per_page=100", token=token)
        comments = self.request("GET", f"/repos/{repository}/pulls/{pull_number}/comments?per_page=100", token=token)
        pull = self.request("GET", f"/repos/{repository}/pulls/{pull_number}", token=token)
        return {"pull_request": pull, "reviews": reviews, "review_comments": comments}

    def open_dependency_pull_requests(self, repository: str, package_name: str, token: str) -> list[dict[str, Any]]:
        query = urllib.parse.quote(f"repo:{repository} is:pr is:open author:app/dependabot {package_name}")
        data = self.request("GET", f"/search/issues?q={query}&per_page=20", token=token)
        return list(data.get("items") or [])

    def app(self, app_jwt: str) -> dict[str, Any]:
        return self.request("GET", "/app", token=app_jwt)
