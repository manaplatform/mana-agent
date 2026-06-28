# Architecture

`mana-agent` is organized as a Python CLI application with supporting service
layers for repository analysis, dependency inspection, code description, and
LLM-driven agent workflows.

## Overview
## Major Components

- `src/mana_agent/commands/`: CLI entry points and output rendering.
- `src/mana_agent/services/`: analysis, ask, report, structure, and security
  services.
- `src/mana_agent/analysis/`: static analysis and chunking helpers.
- `src/mana_agent/dependencies/`: dependency graph support.
- `src/mana_agent/describe/`: repository description service.
- `src/mana_agent/llm/`: prompt chains, agents, tool managers, and workers.
- `src/mana_agent/parsers/`: parser entry points for source files.
- `src/mana_agent/renderers/`: HTML and report rendering.
- `src/mana_agent/tools/`: repository-access and mutation tools used by the agent.
- `src/mana_agent/utils/`: discovery, IO, logging, guards, and tool-run helpers.
- `src/mana_agent/vector_store/`: FAISS-backed vector-store wrapper.

## Data Flow

1. The CLI receives a command such as `analyze`, `ask`, or `chat`.
2. Services coordinate repository discovery, indexing, and analysis.
3. Search and parser layers gather evidence from the codebase.
4. LLM chains and tool managers handle question answering or coding workflows.
5. Renderers and services write artifacts and reports back to disk.

## Repository Layout

```text
src/mana_agent/
  analysis/
  commands/
  config/
  dependencies/
  describe/
  llm/
  parsers/
  renderers/
  services/
  tools/
  utils/
  vector_store/
```

## Related Docs

- [Overview](./01-overview.md)
- [Project Diagram](./07-diagram.md)
- [README](../README.md)
