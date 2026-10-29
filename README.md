# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>

<p align="center">
  <strong>Think beyond prompts. Build beyond boundaries.</strong>
</p>

<p align="center">
  Your AI-powered execution workspace.<br>
  One goal. Multiple agents. Real tools. Verified results.
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10--3.14-3776AB?logo=python&logoColor=white" /></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-22A06B" /></a>
  <img alt="Version" src="https://img.shields.io/badge/Version-v0.6.0-8B5CF6" />
  <a href="https://github.com/manaplatform/mana-agent"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-Mana--Agent-181717?logo=github" /></a>
</p>

<p align="center">
  <a href="#quick-start">Get Started</a> ·
  <a href="#what-can-mana-agent-do">Capabilities</a> 
</p>

---

OpenAI Batch is used only when explicitly requested. Other batching uses Mana's
internal parallel execution. Required Batch tasks pause durably and resume with
results; conversation follow-ups retain both user and assistant context across
frontends. See [Batch execution and recovery](docs/26-openai-batch-api.md).


## What if your AI could do more than answer?

Most AI tools stop at the answer.

You ask a question. They generate a response. Then you're left to figure out what to do next.

**Mana-Agent is built for what happens after the answer.**

It can understand your goals, explore your environment, plan actions, coordinate specialized agents, use real tools, and verify outcomes.

Not just for writing code.

For researching, building, debugging, managing, automating, and solving complex tasks across connected systems.

**You define the destination. Mana-Agent helps navigate the journey.**

## Meet Mana-Agent

Mana-Agent is an open-source, Python-based AI agent orchestration platform combining an autonomous execution loop, specialized agents, persistent task management, supervised tool execution, and extensible integrations.

It brings multiple capabilities into one coordinated environment:

- **Understand:** Analyze repositories, documents, APIs, and available evidence before taking action.
- **Decide:** Select suitable models, agents, tools, and execution strategies.
- **Execute:** Use coding tools, browser automation, external services, and connected systems.
- **Coordinate:** Distribute work among specialized agents while tracking dependencies and progress.
- **Remember:** Reuse authorized context, decisions, and lessons across relevant tasks.
- **Verify:** Evaluate actual outcomes rather than treating tool completion as success.
- **Continue:** Replan, recover, or request human intervention when the goal remains unresolved.

The objective is simple:

**Make AI useful not only for thinking about work, but for carrying work through to a trustworthy result.**

## What can Mana-Agent do?

| Your challenge | What Mana-Agent can help with |
|---|---|
| "I don't understand this codebase." | Explore architecture, dependencies, execution flows, and relevant files. |
| "Something is broken." | Investigate symptoms, identify likely causes, propose fixes, and run verification. |
| "I want to build something complex." | Decompose goals, coordinate coding agents, manage dependencies, and review implementation. |
| "This task has too many steps." | Track progress through the autonomous loop, TaskBoard, and execution supervisor. |
| "I need information from an API." | Inspect API documentation, validate requests, and execute supported operations. |
| "I keep repeating this workflow." | Create automations or teach repeatable actions using Teach Mode. |
| "I want AI to work across tools." | Integrate browser, Git, SSH, email, Telegram, protocols, and external services. |
| "What happened during execution?" | Inspect activity, tools, decisions, errors, checkpoints, and artifacts. |

### One interface. Many kinds of work.

```text
> Explain how authentication works in this repository.

> Find why the latest deployment failed.

> Plan and implement a new feature, then run its tests.

> Read this API documentation and help me call the right endpoint.

> Analyze these files and produce an actionable report.

> Break this project into independent tasks and track progress.

> Review previous mistakes before attempting this operation again.
```

Mana-Agent is designed to support both focused questions and multi-step workflows.

---

## Quick Start

### 1. Install

Requires Python 3.10–3.14.

Recommended installation with pipx:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Or install directly from the repository:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -e .
```

### 2. Configure

```bash
mana-agent --configure
```

Configure your model provider, credentials, model preferences, and supported capabilities.

### 3. Start

```bash
cd /path/to/your/project
mana-agent
```

You're ready.

Start with a question about your repository, ask for an investigation, or describe something you want to accomplish.

**Your first task doesn't need to be complicated.**

Try:

```text
Analyze this project and explain its architecture,
main components, dependencies, and potential risks.
```

### 4. Explore the dashboard

```bash
mana-agent dashboard --root-dir .
```

The optional dashboard provides another way to interact with Mana-Agent, inspect activity, and manage workflows.

---
## License

Mana-Agent is open source under the [MIT License](LICENSE).

---

<p align="center">
  <strong>Mana-Agent</strong><br>
  <em>From intention to execution. From execution to evidence.</em>
</p>

<p align="center">
  <a href="#quick-start">Get Started</a> ·
  <a href="https://github.com/manaplatform/mana-agent">GitHub</a> ·
  <a href="https://github.com/manaplatform/mana-agent/issues">Report an Issue</a> ·
  <a href="docs/">Explore the Docs</a>
</p>

<p align="center">
  <strong>Have a goal in mind? Start with Mana.</strong>
</p>