# Plan Mode and Act Mode in Mana-Agent

Mana-Agent features two primary execution modes: **Plan Mode** and **Act Mode**. These modes ensure that the agent thoroughly analyzes tasks, explores codebase context, identifies dependencies, and produces structured implementation plans before applying any changes.

---

## 1. Overview

| Execution Mode | Purpose | Permissions | Mutations Allowed? |
| :--- | :--- | :--- | :--- |
| **Plan Mode** | Analyze tasks, explore codebase context, map dependencies, assess risks, and formulate structured plans. | Read-Only | **No** (File writes, patches, and destructive commands are blocked) |
| **Act Mode** | Execute implementation steps, apply authorized patches, and run verification. | Read/Write (Authorized) | **Yes** (Subject to approved plans and safety policies) |

---

## 2. Plan Mode Guarantees

In **Plan Mode**, the agent operates under strict read-only guarantees:
- **No File Modifications**: Tools such as `write_file`, `edit_file`, `multi_edit_file`, `apply_patch`, `create_file`, `delete_file`, and document update tools are deterministically blocked at the execution boundary.
- **No Destructive Commands**: Shell execution in Plan Mode permits only non-mutating inspection commands (such as `ls`, `git status`, `git diff`, `git log`, `grep`, `pwd`). Any command attempting file redirection, deletion, or modification fails with `PlanModeError`.
- **Deep Codebase Exploration**: The agent can inspect symbols, call graphs, ASTs, file contents, and project configuration to understand system architecture and dependencies.
- **Structured Plan Generation**: Produces a comprehensive `StructuredPlan` containing:
  - **Objectives**: Clear, verifiable targets.
  - **Context & Dependencies**: Repository context, module structure, and package dependencies.
  - **Affected Files**: Discovered files and candidate modification targets.
  - **Implementation Steps**: Sequential, reviewable steps.
  - **Potential Risks & Mitigations**: Architectural risks, compatibility, or edge cases.
  - **Validation Strategies & Verification Commands**: Exact commands (e.g. pytest, linters) to verify the change.

Plans are persisted in `.mana/plans/<task-slug>.md` for traceability.

---

## 3. Act Mode and User Approval Gating

Transitioning from Plan Mode to Act Mode and executing a proposed plan requires **explicit user approval**:
- **Cannot Execute Unapproved Plans**: Attempting to execute mutations in Act Mode while an unapproved plan is pending prompts the user for approval.
- **Approval Methods**:
  - **CLI**: Pass `--yes` to pre-approve the generated plan, or answer `yes` at the interactive prompt.
  - **Chat / TUI**: Use `/approve` or `/act` to explicitly approve the active plan and switch to Act Mode.
- **Context Preservation**: When switching from Plan Mode to Act Mode, all explored files, identified dependencies, session history, and the active plan are preserved.

---

## 4. Usage

### CLI Commands

Create an implementation plan:
```bash
# Generate and save a plan without implementing
mana-agent plan "Refactor database connection pool" --no-code

# Generate a plan and prompt for approval to execute
mana-agent plan "Refactor database connection pool"

# Pre-approve implementation and switch to Act Mode immediately
mana-agent plan "Refactor database connection pool" --yes
```

### In-Chat Slash Commands & Interactive Workflow

Within interactive chat sessions (`mana-agent chat`):
- `/plan <task>`: Enter Plan Mode, explore repository context, generate a structured markdown plan, save the plan artifact to `.mana/plans/<task-slug>.md`, and prompt for `[accept/reject]`.
  - When **accepted**: The plan is approved, mode switches to **Act Mode**, and implementation starts immediately.
  - When **rejected**: The session safely remains in **Plan Mode** without mutating any files.
- `/mode`: View current execution mode.
- `/mode plan`: Switch to Plan Mode.
- `/mode act`: Switch to Act Mode (requires approval if a plan is pending).
- `/approve` / `/accept`: Explicitly approve the current implementation plan.
- `/reject`: Reject the pending plan and keep the session in Plan Mode.
- `/act`: Approve the active plan and enter Act Mode.

---

## 5. Architectural Alignment with AGENTS.md

- **Rule 07 (Plan Proportionally)**: Plan Mode itself never makes repository changes without an authorized transition to execution.
- **Rule 08 (TaskBoard Authority)**: TaskBoard maintains the authoritative task state, while Plan Mode defines the objectives and acceptance criteria.
- **Rule 11 & 12 (Model Decision Validation)**: Decisions to transition between modes and execute tools are explicitly validated before execution.
- **Rule 21 (Tool-Specific Authority)**: Enforces read-only confinement at the execution boundary rather than relying only on prompt instructions.
- **Rule 40 & 41 (Verification & User Commands)**: Plan Mode defines explicit, user-owned verification commands without silently executing arbitrary verification scripts.
