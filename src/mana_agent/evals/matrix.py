from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Event, Semaphore

from .config import EvalSuite
from .ids import execution_id
from .models import EvalRun
from .runner import EvalRunner, RunSelection


@dataclass(slots=True)
class ExperimentResult:
    experiment_id: str
    runs: list[EvalRun] = field(default_factory=list)
    cancelled: bool = False


def expand_matrix(suite: EvalSuite, *, task_ids: set[str] | None = None, variant_ids: set[str] | None = None) -> list[RunSelection]:
    selections: list[RunSelection] = []
    for task in suite.tasks:
        if task_ids and task.task_id not in task_ids:
            continue
        for variant in suite.variants:
            if variant_ids and variant.variant_id not in variant_ids:
                continue
            for trial in range(1, suite.defaults.trials + 1):
                selections.append(RunSelection(task, variant, trial))
    return selections


class ExperimentRunner:
    def __init__(self, runner: EvalRunner, *, concurrency: int = 1, provider_limits: dict[str, int] | None = None) -> None:
        self.runner = runner
        self.concurrency = max(1, concurrency)
        self.provider_limits = {name: Semaphore(max(1, limit)) for name, limit in (provider_limits or {}).items()}
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def execute(self, suite: EvalSuite, *, task_ids: set[str] | None = None, variant_ids: set[str] | None = None, experiment_id: str | None = None) -> ExperimentResult:
        experiment_id = experiment_id or execution_id("experiment")
        result = ExperimentResult(experiment_id)
        selections = expand_matrix(suite, task_ids=task_ids, variant_ids=variant_ids)

        def execute_one(selection: RunSelection) -> EvalRun | None:
            if self.cancel_event.is_set():
                return None
            semaphore = self.provider_limits.get(selection.variant.main_model)
            if semaphore:
                semaphore.acquire()
            try:
                return self.runner.run(
                    suite=suite, task=selection.task, variant=selection.variant,
                    trial_number=selection.trial_number, experiment_id=experiment_id,
                )
            finally:
                if semaphore:
                    semaphore.release()

        with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="mana-eval") as pool:
            futures = [pool.submit(execute_one, item) for item in selections]
            for future in as_completed(futures):
                run = future.result()
                if run is not None:
                    result.runs.append(run)
        result.runs.sort(key=lambda item: (item.task_id, item.variant_id, item.trial_number))
        result.cancelled = self.cancel_event.is_set()
        return result
