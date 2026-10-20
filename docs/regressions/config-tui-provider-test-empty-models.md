# Configuration TUI provider test with empty model selectors

## Symptom and reproduction

When a user switches the configuration TUI's inference provider to NVIDIA before selecting models and presses **Test provider**, the model selectors may still be blank. The TUI's `_collect()` passes those empty values to `ConfigurationDraft.set_models()`, which previously called `qualify_model_id()` unconditionally and raised `ValueError: Provider and model ID are required.` before the provider connectivity check could run.

## Root cause

`set_models()` treated transiently empty model selectors as valid model IDs. Provider testing happens before the model catalog is populated, so this workflow state is expected; it is not a malformed model identifier that should escape as an exception.

## Permanent fix and regression prevention

`ConfigurationDraft.set_models()` now always records the provider selection, but only replaces model settings when both required selectors (high-reasoning and embedding) contain values. This keeps the last valid model configuration intact while the provider catalog is being tested or selectors are only partially populated. Fully selected models continue to be qualified and stored as before.

Regression tests cover all-empty selectors, a partially populated selector set, and normal NVIDIA model qualification.

**Do not reintroduce:** Do not call `qualify_model_id()` for an empty or incomplete model selection. Empty selectors are a normal pre-catalog TUI state; provider selection must be recorded independently from model selection.

## Verification

Run `./venv/bin/pytest tests/test_configuration_session.py -q` to check these regression cases. The test status is determined by the run output; this note does not imply a passing result.
