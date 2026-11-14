"""User-run clean installations and compatibility evidence. Never imported by runtime."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", action="append", required=True)
    parser.add_argument("--extras", default="dev,full")
    parser.add_argument("--all-extras", action="store_true", help="Isolate every declared extra; Linux desktop input is opt-in")
    parser.add_argument("--include-linux-desktop", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-suite", action="store_true")
    parser.add_argument("--build", action="store_true")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = options.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError("Use an output directory outside the checkout")
    output.mkdir(parents=True, exist_ok=True)
    optional = tomllib.loads((root / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    groups = [options.extras]
    if options.all_extras:
        groups = ["dev", *[f"dev,{extra}" for extra in optional if extra != "dev" and
            (extra != "teach-desktop" or os.name == "nt" or __import__("sys").platform == "darwin" or options.include_linux_desktop)]]
    results = []
    for index, interpreter in enumerate(options.python):
        executable = shutil.which(interpreter)
        if executable is None:
            results.append({"interpreter": interpreter, "status": "blocked", "reason": "Interpreter unavailable"})
            continue
        for group_index, extras in enumerate(groups):
            directory = output / f"python-{index}-extras-{group_index}"
            if directory.exists():
                raise ValueError(f"Refusing to reuse verification environment {directory}")
            directory.mkdir()
            environment = {key: value for key, value in os.environ.items() if
                not key.endswith(("API_KEY", "API_TOKEN")) and not key.startswith("MANA_") and key != "PYTHONPATH"}
            environment["MANA_HOME"] = str(directory / "mana-home")
            python = directory / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            dist = directory / "dist"
            commands = [([executable, "-m", "venv", str(directory / "venv")], directory),
                ([str(python), "-m", "pip", "install", "--upgrade", "pip"], directory),
                ([str(python), "-m", "pip", "install", "build", "twine"], directory),
                ([str(python), "-m", "build", "--wheel", "--outdir", str(dist), str(root)], directory),
                ([str(python), "-m", "pip", "install", "WHEEL_PLACEHOLDER"], directory),
                ([str(python), "-m", "pip", "check"], directory),
                ([str(python), str(root / "scripts/audit_native_dependencies.py"), "--extras", extras,
                    "--output", str(directory / "dependency-audit.json")], directory),
                ([str(python), "-m", "mana_agent.commands.cli", "--help"], directory),
                ([str(python), "-m", "pytest", "-q", *([] if options.full_suite else ["tests/llm", "tests/test_model_capabilities.py",
                    "tests/test_model_routing.py", "tests/test_repository_tools.py", "tests/test_project_search.py"])], root)]
            if options.build:
                commands.extend([([str(python), "-m", "build", "--sdist", "--outdir", str(dist), str(root)], directory),
                                 ([str(python), "-m", "twine", "check", "DIST_PLACEHOLDER"], directory)])
            steps = []
            for step, (command, cwd) in enumerate(commands):
                log = directory / f"step-{step}.log"
                if "WHEEL_PLACEHOLDER" in command:
                    wheels = list(dist.glob("*.whl"))
                    if len(wheels) != 1:
                        raise RuntimeError("Expected one built wheel")
                    command[-1] = str(wheels[0]) + f"[{extras}]"
                if "DIST_PLACEHOLDER" in command:
                    command = command[:-1] + [str(path) for path in sorted(dist.iterdir())]
                try:
                    with log.open("w") as handle:
                        completed = subprocess.run(command, cwd=cwd, env=environment, stdout=handle, stderr=subprocess.STDOUT, timeout=5400)
                    code = completed.returncode
                except (subprocess.TimeoutExpired, OSError) as exc:
                    log.write_text(f"{type(exc).__name__}: {exc}\n")
                    code = 124
                steps.append({"command": command, "cwd": str(cwd), "exit_code": code, "log": str(log)})
                if code:
                    break
            results.append({"interpreter": interpreter, "extras": extras,
                "status": "passed" if len(steps) == len(commands) and not steps[-1]["exit_code"] else "failed", "steps": steps})
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return int(any(row["status"] != "passed" for row in results))

if __name__ == "__main__":
    raise SystemExit(main())
