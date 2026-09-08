# Coding Agent Skill

## When to use

Use this skill when a task requires tools, patching, file writes, code edits, or coding-agent execution.

## Rules

- Read relevant files before editing.
- Preserve user changes and avoid unrelated rewrites.
- Use existing tool contracts and queue ownership.
- Verify changed behavior with focused tests.
- Stop endless loops with clear blocked reasons and saved evidence.

## Verification

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_codex_integration.py tests/test_agent_work_queue.py -q
```

