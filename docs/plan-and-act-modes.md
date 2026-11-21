# AUTO, DEBUG, and PLAN in Mana-Agent

Describe the repository change or problem in ordinary chat. The gateway model selects **AUTO** for implementation or **DEBUG** for investigation and repair. Use **PLAN** when you want to review a revision before implementation. The global logging flag is still called `--debug`.

Mana reads the original request, relevant conversation context for follow-ups, and any saved answers before deciding what to do. The workflow asks only for missing information that you need to provide and that affects whether work can continue. It uses repository investigation for questions that the checkout can answer.

When the workflow requests an answer, it pauses at its saved checkpoint. The TUI opens an answer form and interactive console chat asks for each answer. After submission, Mana continues the same task with its existing evidence. Choose **Later** or press Enter at a console answer prompt to leave the task pending in chat. You can also submit the full set with `/inbox answer <item-id> '<JSON answers keyed by question ID>'`, or use `mana-agent inbox answer <item-id> --answer '<JSON answers keyed by question ID>'` from a separate terminal.

MainAgent assigns the Coding task after selecting a mode. The realtime Internal backend chooses reads, searches, edits, diagnostics, and verification directly within that assignment. Codex selects its own tools through app-server. Explicitly requested stand-alone repository analysis remains read-only.

Realtime Coding can call `investigate_repository` for up to four scoped operations or `integrate_repository_evidence` to retrieve current, ranked Evidence Board records and conflicts. Ordinary reads also save source-backed observations. Read a known file directly and use a line range when only a section is relevant or the file exceeds the available budget. Read results include the covered lines, total lines, file hash, and a continuation line when the output budget truncates the range.

An `ls` decision takes the tool name, level 1, and a reason. It lists immediate directories from the checkout root. Use `list_files` with a directory path or targeted glob to inspect a subdirectory. Use `repo_search` with a nonblank literal `query` to find text in the selected files. A bounded `inspect` request at levels 1–3 leaves `initial_repository_scope` empty. An audit value is reserved for an initial level-4 discovery for an explicit user-requested audit or architecture-wide investigation.

PLAN and staged workflows execute `inspect` batches through the current task's read-only tools. Every entry is validated before dispatch. Identical reads run once and count once against the shared allowance. The Autonomous Loop retains its no-progress and budget limits.

## Implement a change

With the realtime Internal backend, AUTO can start with an edit, read, search, or diagnostic. It can maintain a working breakdown during a large change and verify incrementally. Production capability changes require connected source references and passing isolated verification. Tool and action approvals still apply. The existing staged workflow handles external Codex, explicit Batch, and saved plan revisions.

When Codex worktree isolation is enabled, AUTO and DEBUG repairs inspect, patch, and verify in the task's assigned worktree. The result reports the verified worktree path for review. The primary checkout retains the user's changes.

When a verification command fails, native Coding receives the exit code and output in the same loop. It may inspect the cause, edit, and rerun the required checks. Verification is tied to the versions of the changed files, so later changes require a fresh check. Unavailable verification environments produce a blocked result with the task's evidence preserved.

The gateway checks permissions and leases throughout execution. Duplicate native observations reuse their earlier result in conversation context. If actions stop producing new information or changes, Coding receives a bounded opportunity to pick a useful action or report the blocker. Use `/task resume <task-id>` to recover a stopped task with reconciled actions.

A new scoped investigation uses `recovery: null` and lists the operations to execute. An explicit recovery for a legacy investigation matches the saved child and checkpoint. It imports current evidence and returns any remaining question to Coding.

## Plan a change

Explicit PLAN uses the same Coding-owned repository observations while preparing a revision. Task and evidence writes are durable, while repository writes remain blocked until the exact plan revision is approved. New revisions cite Evidence Board references; old approved revisions retain their original digest.

- In chat or the TUI, use `/plan <request>`. Use `/plan` by itself to inspect the active task or start planning your next message. `/mode plan` and `mana-agent chat --planning-mode` enter the same workflow. Use `/mode auto` to return to ordinary routing after resolving an active plan. A proposal requested in prose is a read-only response.
- From a terminal, use `mana-agent plan 'Upgrade the connection pool'`. Use `--no-code` for a draft with no execution approval. `--no-save` skips the Markdown export. For `--output`, select a path outside the target checkout. The default export resides under Mana's repository state directory.
- If the plan depends on a missing choice only you can make, Mana asks through the durable inbox. Select an option or enter a custom answer when available.
- Review the exact requirements, boundaries, dependencies, file operations, and checks. Use the TUI approval controls or `/approve <item-id>` (`/accept` and `/act` are aliases in that context). In a standalone invocation, use the displayed review prompt or run `mana-agent inbox approve <item-id>`. A bare `--yes` or a Full Access session does not approve a generated revision.
- Use `/revise <change>` to update the plan. To correct a stored clarification answer, supply a JSON object keyed by its question ID. A new revision requires a new approval. Use `/reject` or the existing cancellation controls to stop the task.

You can paste a multi-paragraph request after `/plan`. Mana forwards the request
as one text argument, preserving internal line breaks and punctuation in ordinary
prose. You can still quote the whole request as a single shell-style argument.

In the TUI, submitting `/plan` clears the composer immediately. Plan previews and
clarification questions appear in dialogs with scrollable content and a fixed
row area for the decision buttons. Choose **Later** to return to chat, then use
`/plan` to reopen the pending dialog. After you submit a decision, the
conversation shows activity for the originating turn and the eventual result.

If a model response fails validation, Mana requests one correction using the same model and response format. If the correction fails, Mana shows the failed field or check. Run `/plan` to see the blocked task, then use the displayed `/task resume <task-id>` command to retry planning.

The model returns typed records for file operations and clarification choices. Mana validates those records before using them. Saved revisions and their approval digests use the existing plan format.

TaskBoard records step progress. The Execution Supervisor owns the active attempt and lease. The human inbox records the reviewer, timestamp, and digest of the approved revision. Ordinary mode switches and free text cannot substitute for an inbox response. Completed steps are kept when their scope and evidence still match. A patch with an uncertain outcome requires reconciliation before retry. `ACT` and `EDIT` remain compatibility names for authorized execution phases and tool policy. They are not model-selected workflows.

Each PLAN patch proposal is read-only. The internal engine uses registered read tools, and external Codex uses a read-only app-server turn with networking disabled. Mana checks the exact step, current scope, file operations, checkout state, and patch content before the existing Tool Runner and action policy apply a mutation. Scope changes from live steering return the task to review. Git publishing and mutations to external services keep their existing approval flows.

If the action policy requires a separate approval for a destructive patch, review that action in the inbox and resume the plan. Mana reuses the exact reviewed patch and checks the plan revision again before applying it.

Patch proposals and application use the assigned checkout. In the staged workflow, verification runs the reviewed argv in Docker against a disposable source copy without `.git`, host secrets, or sandbox networking. Use project checks for that environment and supply a locally available image with the required dependencies. A step with no image uses the local Docker provider default. If an automatic repair encounters an unavailable check before its first patch, Coding can revise the internal plan using saved evidence and move the required passing checks after the edits. An explicitly approved PLAN retains its reviewed order. If verification is still unavailable after the patch, Mana reports the applied files and the pending check. Once the verifier is ready, `/task resume <task-id>` continues from saved progress.

## Debug a failure

Describe the failure in ordinary chat. A stack trace, failing command, or affected file and line helps Mana select the first action. Ask it to diagnose the problem or to diagnose and fix it. Realtime Internal DEBUG can read the identified code, apply the smallest supported repair, and iterate on a focused build or check in the assigned checkout. The Textual execution stream shows failed builds, budget renewals, and repeated actions without progress. [Adaptive coding runtime](adaptive-coding-runtime.md)

A reproduction command that expects failure requires a literal output marker that identifies the intended failure. If the evidence cannot establish that diagnostic, the repair plan can record the uncertainty and use a regression check that expects to pass. The same decision rules apply when repository inspection reaches its no-progress limit and the workflow must use saved observations.
