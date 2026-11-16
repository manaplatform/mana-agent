# AUTO, DEBUG, and PLAN in Mana-Agent

Describe the repository change or problem in ordinary chat. The gateway model selects **AUTO** for implementation or **DEBUG** for investigation and repair. Use **PLAN** when you want to review a revision before implementation. The global logging flag is still called `--debug`.

Mana reads the original request, relevant conversation context for follow-ups, and any saved answers before deciding what to do. The workflow asks only for missing information that you need to provide and that affects whether work can continue. It uses repository investigation for questions that the checkout can answer.

When the workflow requests an answer, it pauses at its saved checkpoint. The TUI opens an answer form and interactive console chat asks for each answer. After submission, Mana continues the same task with its existing evidence. Choose **Later** or press Enter at a console answer prompt to leave the task pending in chat. You can also submit the full set with `/inbox answer <item-id> '<JSON answers keyed by question ID>'`, or use `mana-agent inbox answer <item-id> --answer '<JSON answers keyed by question ID>'` from a separate terminal.

When Mana needs repository evidence, the main agent delegates one focused question to a read-only coding session on the selected chat model. The model uses native tools for related searches and file reads. The main agent saves the observations on TaskBoard, then decides whether to diagnose, draft, ask a question, or stop. Each investigation has one bounded allowance and reports any gaps it could not resolve.

## Implement a change

AUTO starts with scoped repository observations, records an internal plan and its verification requirements on TaskBoard, and executes the plan through the configured CodingAgent. The gateway projects the workflow on the same task through integration, testing, and review. Runtime capabilities require evidence from the existing feature integration coordinator before they are reported as complete. An internal plan has no separate PLAN approval step. Tool and action approvals still apply.

When Codex worktree isolation is enabled, AUTO and DEBUG repairs inspect, patch, and verify in the task's assigned worktree. The result reports the verified worktree path for review. The primary checkout retains the user's changes.

DEBUG and AUTO can switch when the model cites new observed evidence on the same task. Scope, completed steps, and execution permissions carry forward. When a required verification command runs and fails, an automatic workflow can revise its plan once within the accepted file operations and rerun that command. Other failures and unavailable verification environments produce a blocked result with the task's evidence preserved.

## Plan a change

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

Verification commands run with the reviewed argv against a disposable source copy, without host secrets or sandbox networking. Use a locally available Docker image that contains the project's verification dependencies. A step can name that image in the plan. With no image specified, the configured local Docker provider default applies. If the environment or command is unavailable, the task shows verification as incomplete. Fix the environment and use `/task resume <task-id>` to retry from the saved progress.

## Debug a failure

Describe the failure in ordinary chat. A stack trace, failing command, or affected file and line helps Mana select the first observation. Ask it to diagnose the problem or to diagnose and fix it. A repair starts with causal evidence and ends with focused verification. External dependency research uses the existing coding research settings and tool permissions. The Activity panels show the current investigation, justified scope expansions, patch work, and verification results.
