# AUTO, DEBUG, and PLAN in Mana-Agent

Describe the repository change or problem in ordinary chat. The gateway model selects **AUTO** for implementation or **DEBUG** for investigation and repair. Use **PLAN** when you want to review a revision before implementation. The global logging flag is still called `--debug`.

Mana reads the original request, relevant conversation context for follow-ups, and any saved answers before deciding what to do. The workflow asks only for missing information that you need to provide and that affects whether work can continue. It uses repository investigation for questions that the checkout can answer.

When the workflow requests an answer, it pauses at its saved checkpoint. The TUI opens an answer form and interactive console chat asks for each answer. After submission, Mana continues the same task with its existing evidence. Choose **Later** or press Enter at a console answer prompt to leave the task pending in chat. You can also submit the full set with `/inbox answer <item-id> '<JSON answers keyed by question ID>'`, or use `mana-agent inbox answer <item-id> --answer '<JSON answers keyed by question ID>'` from a separate terminal.

Repository investigation and understanding are independently supervised TaskBoard tasks in every mode. MainAgent queries the persistent Repository Evidence Board before choosing more work. Current findings can be shared across tasks; missing, contradictory, or stale findings can cause a focused investigation task. Coding waits only on its own dependencies, and unrelated tasks can continue. Investigation completion requires satisfying evidence requirements, rather than counting reads.

Investigation uses the configured coding engine in read-only mode, including external Codex when selected. Native reads use the registered ToolRunner boundary; Codex retains its inner loop and provider-owned provenance. Findings remain inferences or hypotheses until supported by actual observations. Scoped source validation, content hashes, and task/attempt identities accompany persisted evidence. There is no mandatory investigation stage. See [evidence contracts and recovery](regressions/repository-evidence-orchestration.md).

An `ls` decision takes the tool name, level 1, and a reason. It lists immediate directories from the checkout root. Use `list_files` to find files by path or glob. Use `repo_search` with a nonblank literal `query` to find text in the selected files.

AUTO and DEBUG execute `inspect` batches through the parent task's read-only tools. Source-backed results are saved to the Evidence Board, and exact reads reuse current observations. The Autonomous Loop retains its no-progress and budget limits. An explicit `investigate` decision creates a separate supervised assignment when the question needs delegated analysis.

## Implement a change

AUTO can draft as its first action when the request and available context establish the files, intended change, and focused check. The model can also request a specific observation that would change its plan. AUTO records the internal plan and verification requirements on TaskBoard and executes through the configured CodingAgent. The gateway projects the workflow on the same task through integration, testing, and review. Runtime capabilities require evidence from the existing feature integration coordinator before they are reported as complete. An internal plan has no separate PLAN approval step. Tool and action approvals still apply.

When Codex worktree isolation is enabled, AUTO and DEBUG repairs inspect, patch, and verify in the task's assigned worktree. The result reports the verified worktree path for review. The primary checkout retains the user's changes.

DEBUG and AUTO can switch when the model cites new observed evidence on the same task. Scope, completed steps, and execution permissions carry forward. When a required verification command fails, the model can revise its plan within the accepted file operations, retain required checks, and rerun the failed command. Unavailable verification environments produce a blocked result with the task's evidence preserved.

Mode decisions validate citations, action availability, repair scope and checks, and evidence-resume references before the selected model's single correction. The gateway rechecks current evidence and authority before execution. Automatic write workflows reserve shared budget for the first patch and focused verification. When a native patch proposal reaches its read or model budget, or the read no-progress limit, it has a final response turn to submit the patch or identify the concrete missing fact that prevents it.

## Plan a change

Explicit PLAN can create read-only investigation and understanding tasks through the selected coding engine. Task and evidence writes are durable, while repository writes remain blocked until the exact plan revision is approved. New revisions cite Evidence Board references; old approved revisions retain their original digest.

- In chat or the TUI, use `/plan <request>`. Use `/plan` by itself to inspect the active task or start planning your next message. `/mode plan` and `mana-agent chat --planning-mode` enter the same workflow. Use `/mode auto` to return to ordinary routing after resolving an active plan. A proposal requested in prose is a read-only response.
- From a terminal, use `mana-agent plan 'Upgrade the connection pool'`. Use `--no-code` for a draft with no execution approval. `--no-save` skips the Markdown export. For `--output`, select a path outside the target checkout. The default export resides under Mana's repository state directory.
- If the plan depends on a missing choice only you can make, Mana asks through the durable inbox. Select an option or enter a custom answer when available.
- Review the exact requirements, boundaries, dependencies, file operations, and checks. Use the TUI approval controls or `/approve <item-id>` (`/accept` and `/act` are aliases in that context). In a standalone invocation, use the displayed review prompt or run `mana-agent inbox approve <item-id>`. A bare `--yes` or a Full Access session does not approve a generated revision.
- Use `/revise <change>` to update the plan. To correct a stored clarification answer, supply a JSON object keyed by its question ID. A new revision requires a new approval. Use `/reject` or the existing cancellation controls to stop the task.

If a model response fails validation, Mana requests one correction using the same model and response format. If the correction fails, Mana shows the failed field or check. Run `/plan` to see the blocked task, then use the displayed `/task resume <task-id>` command to retry planning.

The model returns typed records for file operations and clarification choices. Mana validates those records before using them. Saved revisions and their approval digests use the existing plan format.

TaskBoard records step progress. The Execution Supervisor owns the active attempt and lease. The human inbox records the reviewer, timestamp, and digest of the approved revision. Ordinary mode switches and free text cannot substitute for an inbox response. Completed steps are kept when their scope and evidence still match. A patch with an uncertain outcome requires reconciliation before retry. `ACT` and `EDIT` remain compatibility names for authorized execution phases and tool policy. They are not model-selected workflows.

Each coding proposal is read-only. The internal engine uses registered read tools, and external Codex uses a read-only app-server turn with networking disabled. Mana checks the exact step, current scope, file operations, checkout state, and patch content before the existing Tool Runner and action policy apply a mutation. Scope changes from live steering return the task to review. Git publishing and mutations to external services keep their existing approval flows.

If the action policy requires a separate approval for a destructive patch, review that action in the inbox and resume the plan. Mana reuses the exact reviewed patch and checks the plan revision again before applying it.

Verification commands run with the reviewed argv against a disposable source copy, without host secrets or sandbox networking. Use a locally available Docker image that contains the project's verification dependencies. A step can name that image in the plan. With no image specified, the configured local Docker provider default applies. The final chat reply reports applied changes and whether verification passed, failed, or is pending. Coding Activity shows command details. If Docker or the required image is unavailable, prepare the local environment and use the displayed `/task resume <task-id>` command to continue from saved progress.

## Debug a failure

Describe the failure in ordinary chat. A stack trace, failing command, or affected file and line helps Mana select the first observation. Ask it to diagnose the problem or to diagnose and fix it. A repair starts with causal evidence and ends with focused verification. External dependency research uses the existing coding research settings and tool permissions. The Activity panels show the current investigation, justified scope expansions, patch work, and verification results.

A reproduction command that expects failure requires a literal output marker that identifies the intended failure. If the evidence cannot establish that diagnostic, the repair plan can record the uncertainty and use a regression check that expects to pass. The same decision rules apply when repository inspection reaches its no-progress limit and the workflow must use saved observations.
