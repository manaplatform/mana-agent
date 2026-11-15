# Plan, Act, and Debug in Mana-Agent

Use **Plan** when you want to review the proposed changes before work begins. In ordinary chat, the model can select **Debug** for a reported repository problem. Adding a bug report to a Plan request keeps it in the plan approval workflow. The global logging flag is still called `--debug`.

## Plan a change

- In chat or the TUI, use `/plan <request>`. Use `/plan` by itself to inspect the active task or start planning your next message. `/mode plan` also selects planning for subsequent messages. Use `/mode auto` to return to ordinary routing after resolving an active plan.
- From a terminal, use `mana-agent plan 'Upgrade the connection pool'`. Use `--no-code` for a draft with no execution approval. `--no-save` skips the Markdown export. For `--output`, select a path outside the target checkout. The default export resides under Mana's repository state directory.
- If a decision would change the implementation, Mana asks through the durable inbox. Select an option or enter a custom answer when available. In console chat, reply directly to one question or use `/inbox answer <item-id> '<JSON answers keyed by question ID>'` for a group.
- Review the exact requirements, boundaries, dependencies, file operations, and checks. Use the TUI approval controls or `/approve <item-id>` (`/accept` and `/act` are aliases in that context). In a standalone invocation, use the displayed review prompt or run `mana-agent inbox approve <item-id>`. A bare `--yes` or a Full Access session does not approve a generated revision.
- Use `/revise <change>` to update the plan. To correct a stored clarification answer, supply a JSON object keyed by its question ID. A new revision requires a new approval. Use `/reject` or the existing cancellation controls to stop the task.

TaskBoard records step progress. The Execution Supervisor owns the active attempt and lease. The human inbox records the reviewer, timestamp, and digest of the approved revision. Ordinary mode switches and free text cannot substitute for an inbox response. Completed steps are kept when their scope and evidence still match. A patch with an uncertain outcome requires reconciliation before retry.

Each coding proposal is read-only. The internal engine uses registered read tools, and external Codex uses a read-only app-server turn with networking disabled. Mana checks the exact step, current scope, file operations, checkout state, and patch content before the existing Tool Runner and action policy apply a mutation. Scope changes from live steering return the task to review. Git publishing and mutations to external services keep their existing approval flows.

If the action policy requires a separate approval for a destructive patch, review that action in the inbox and resume the plan. Mana reuses the exact reviewed patch and checks the plan revision again before applying it.

Verification commands run with the reviewed argv against a disposable source copy, without host secrets or sandbox networking. Use a locally available Docker image that contains the project's verification dependencies. A step can name that image in the plan. With no image specified, the configured local Docker provider default applies. If the environment or command is unavailable, the task shows verification as incomplete. Fix the environment and use `/task resume <task-id>` to retry from the saved progress.

## Debug a failure

Describe the failure in ordinary chat. A stack trace, failing command, or affected file and line helps Mana select the first observation. Ask it to diagnose the problem or to diagnose and fix it. A repair starts with causal evidence and ends with focused verification. External dependency research uses the existing coding research settings and tool permissions. The Activity panels show the current investigation, justified scope expansions, patch work, and verification results.
