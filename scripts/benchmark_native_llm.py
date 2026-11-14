"""Repeatable offline import/search/context measurements; run once per checkout."""
from __future__ import annotations
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WORKER = r'''
import json, time, tracemalloc
tracemalloc.start()
started = time.perf_counter()
import mana_agent.commands.cli
import_seconds = time.perf_counter() - started
peak_bytes = tracemalloc.get_traced_memory()[1]
from mana_agent.tools.repository import repo_search
from pathlib import Path
root = Path.cwd()
implementation = __import__("sys").argv[1]
started = time.perf_counter()
for _ in range(10):
    repo_search(root, query="native", glob="src/mana_agent/**/*.py", limit=20)
dispatch_seconds = (time.perf_counter() - started) / 10
messages = [{"role": "user", "content": "context" * 1000} for _ in range(20)]
started = time.perf_counter()
for _ in range(100):
    json.dumps(messages, separators=(",", ":"))
serialization_seconds = (time.perf_counter() - started) / 100
metrics = {"import_seconds": import_seconds, "peak_python_bytes": peak_bytes,
    "repository_dispatch_seconds": dispatch_seconds, "context_serialization_seconds": serialization_seconds}
if implementation == "native":
    from mana_agent.llm.providers.openai import OpenAIProvider
    from mana_agent.llm.messages import UserMessage
    from mana_agent.llm.types import InferenceRequest, StreamEvent
    adapter = OpenAIProvider(provider="openai", api_key="offline", base_url="https://unused.invalid", endpoint="chat_completions")
    request = InferenceRequest(provider="openai", model="gpt-4.1-mini", messages=[UserMessage("sample")])
    construct = lambda: adapter.encode(request)
    stream_part = lambda text: StreamEvent(type="text.delta", request_id="fixture", sequence=1, delta=text)
    serialize = lambda part: part.model_dump_json()
else:
    # This branch runs ONLY in a separate historical-checkout environment. It
    # never installs or imports the retired framework into the native environment.
    import importlib
    legacy = importlib.import_module("mana_agent.multi_agent.runtime.compatibility")
    adapter = legacy.create_chat_model(provider="openai", api_key="offline", model="gpt-4.1-mini", base_url="https://unused.invalid")
    construct = lambda: adapter._get_request_payload(messages[:1])
    legacy_messages = importlib.import_module("langchain_core.messages")
    stream_part = lambda text: legacy_messages.AIMessageChunk(content=text)
    serialize = lambda part: part.model_dump_json()
started = time.perf_counter()
for _ in range(100):
    construct()
metrics["request_construction_seconds"] = (time.perf_counter() - started) / 100
started = time.perf_counter()
for _ in range(100):
    parts = [stream_part("fixture token") for _ in range(50)]
    encoded = [serialize(part) for part in parts]
metrics["stream_event_assembly_seconds"] = (time.perf_counter() - started) / 100
metrics["streaming_network_latency"] = {"unavailable": "Requires separately authorized live transport measurements"}
print(json.dumps(metrics))
'''

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--label", required=True)
    parser.add_argument("--implementation", choices=["native", "baseline"], default="native")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    options = parser.parse_args()
    runs = []
    with tempfile.TemporaryDirectory(prefix="mana-native-benchmark-") as temporary:
        environment = {**os.environ, "MANA_HOME": temporary, "PYTHONPATH": str(options.source_root / "src")}
        for key in list(environment):
            if key.endswith(("API_KEY", "API_TOKEN")):
                environment.pop(key)
        for _ in range(options.runs):
            started = time.perf_counter()
            completed = subprocess.run([options.python, "-c", WORKER, options.implementation], cwd=options.source_root,
                env=environment, capture_output=True, text=True, timeout=120, check=True)
            metrics = json.loads(completed.stdout.splitlines()[-1])
            metrics["startup_seconds"] = time.perf_counter() - started
            runs.append(metrics)
    medians = {key: statistics.median(row[key] for row in runs) if isinstance(runs[0][key], (int, float)) else runs[0][key] for key in runs[0]}
    result = {"label": options.label, "implementation": options.implementation, "runs": len(runs), "medians": medians, "samples": runs}
    if options.compare:
        baseline = json.loads(options.compare.read_text())
        result["ratios_to_baseline"] = {key: value / baseline["medians"][key] for key, value in medians.items()
            if isinstance(value, (int, float)) and isinstance(baseline["medians"].get(key), (int, float)) and baseline["medians"][key]}
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
