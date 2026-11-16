# Intent-driven repository discovery

## Symptom and cause

A localized request such as “Add sudo support to command execution” made the
agent inspect unrelated components before editing. Some of that work happened
even if the model selected the right files:

- The AskAgent search binding first called the compatibility index service.
  That service now searches live files and did not receive the selected glob or
  paths. It searched each configured index, then fell through to a second
  search if it found nothing. Directory mode also searched every index before
  invoking the tool loop. The classic repository routes sent the full question
  to separate whole-checkout search functions.
- The legacy coding queue synthesized a repository search from the complete
  request. Repository tools walked the checkout before applying literal
  directory prefixes. AST tools had no file scope, and a batch of text
  queries read the same candidates repeatedly.
- Coding turns automatically prepended the last project analysis, whether or
  not it was relevant or current. A prompt builder also selected complete skill
  bodies by keyword. Separately, ToolRunner took before-and-after file snapshots
  for commands already recognized as strictly read-only. Earlier tests
  observed the public tool result, so they could pass without seeing these
  extra filesystem operations.

## Corrected behavior

The routing or coding model supplies a focused operation as part of its
existing decision. A known file can go straight to a read. Searches use
repository-relative files/directories or a targeted glob. A literal directory
prefix prunes the traversal itself; a distinctive filename glob can enumerate
names if the owning directory is unknown, but opens only matching files.
A plain directory listing only visits the immediate children. Text and symbol
searches and grouped searches share the same scope checks. Batch text queries
with the same roots share enumeration and source reads.

Unrestricted search requires a model-selected reason grounded in an explicit
audit, architecture-wide impact, systemic evidence, or a gap narrower searches
could not resolve. Plan and Debug enforce progression by checking evidence
references, except when the initial decision explicitly identifies an audit or
architecture-wide request. A miss is an observation for the existing model
loop; the tool never widens the search by itself. Scoped legacy work orders
enforce their parent's repository boundary before transport.

Main delegates technical inspection to Coding. ToolRunner receives the selected
operation and enforces its scope; it does not choose additional operations. The
native coding runner reuses exact operations over named files within the bound
task and checkout while their file versions are unchanged. Directory results
remain observations in the active conversation; they are not put in a reusable
cache without a complete version. Known paths and scoped evidence can be passed
through the existing queue without another general search.

Ordinary turns do not load saved project analysis. The explicit analysis command
still builds reports. Repository rules remain available, and skill bodies are
loaded by selection rather than task-text matching. Internal Coding and Plan /
Debug support bounded line reads. Commands classified as strictly read-only skip
mutation snapshots; potentially mutating commands still use the existing
observation, approval, lock and receipt paths. TaskBoard and Execution
Supervisor keep their existing roles. External Codex continues to own its inner
tool loop; Mana no longer prepends unrelated project analysis to the handoff.

## Investigation scope correction in Plan and Debug

A correctly shaped inspection decision could still block the task with
`Investigation expansion requires observed evidence from the narrower scope`.
In the reported incident, the persisted investigation had three level-1
observations. An ordinary next operation could use level 2 with a valid
citation. The rejected selection's level and IDs were unavailable. The prompt
described broadening generally, while the gateway checked the next-level and
citation rules only after inference. The prompt also showed detailed results
for only the most recent 12 observations, so an older observation at the
highest level could be absent.

Each decision now gets the highest observed level, next permitted ordinary
level, and IDs and levels of all stored observations. Detailed results still
use the bounded history window. Native inference uses that same task-state
snapshot to validate inspection decisions during its existing correction
attempt. The correction distinguishes skipped levels, missing IDs, unknown
IDs, and missing citations at the highest observed level. The gateway checks
the scope again before executing the selected read-only operation.

## Do not reintroduce

- Never map the whole request to a source search or run a compatibility index
  search before honoring model-selected paths.
- Never reset a delegated task's discovery scope, or start a full tree walk for
  a scoped glob. A failed search may inform a separate model decision.
- Never treat cached observations or a search rationale as write permission.
  Reuse file evidence only in its permitted checkout and at its observed version.
- Never collect mutation snapshots around a strictly read-only command solely
  because it uses the command transport.
- Never hide the scope ceiling and evidence IDs from the decision model or
  invent observations to satisfy an expansion gate.

## Verification

Run from the repository root in the development environment:

    python -m pytest -q tests/test_intent_driven_discovery.py tests/test_ask_agent.py tests/test_ask_service.py tests/test_ask_entry_router.py tests/gateway/test_debug_plan_mode_contract.py

The sudo fixture uses scripted tool selections and instruments actual file
opens and directory walks. A replay against the Mana-Agent checkout opened only
the shell executor, transactional action policy, and shell executor tests. It
did not walk the repository root. The replay does not implement sudo support
or call a live model. For interactive acceptance, submit the example in a disposable checkout,
inspect the execution trace, and stop before applying any proposed sudo change.

For Plan and Debug scope correction, run

    venv/bin/python -m pytest -q tests/llm/test_engine.py tests/gateway/test_debug_plan_mode_contract.py -k 'investigation_scope or unjustified_expansion or repository_audit'

The fixtures cover the four correction reasons, references to older
observations, rejection before a broader tool runs, and the explicit audit
path. The agent did not run these tests or make a live provider call.
