# CLAUDE.md — Project Working Agreement

This file defines the conventions and workflow for this project. These rules apply to all work done in this repository.

---

## Journal

Maintain a [Journal.md](Journal.md) file in the project root. For every interaction:

- Record the user's prompt (summarized or verbatim as appropriate).
- Record the outcome: what was done, what was created/changed, and any key decisions made.
- Append new entries; never overwrite old ones.
- This journal is our shared, persistent record of context across sessions.

---

## Planning Requirement

For every **major task** (new features, architectural changes, multi-file refactors, new systems), propose a written plan for user review **before beginning implementation**. The plan should include:

- Objective and scope
- Proposed approach / architecture
- Files to be created or modified
- Test plan (see below)
- Any open questions or trade-offs

Do **not** require a plan for minor tasks (typo fixes, small tweaks, single-line changes, clarifying questions).

Wait for explicit approval before proceeding with implementation of a major task.

---

## Test-Driven Development

All test code lives in the [tests/](tests/) folder.

- Write tests **before or alongside** implementation (TDD style).
- For every major task, include a **test plan** in the proposal that lists what will be tested and how.
- Run tests and record results (pass/fail counts, any failures) in the Journal.
- If a task does not lend itself to automated testing, note why in the Journal.
- Tests should be organized to mirror the structure of the source code they cover.

---

## Documentation

All documentation lives in the [docs/](docs/) folder.

- Create or update documentation for any new system, module, or non-obvious behavior.
- Documentation should be written for a future developer unfamiliar with the current session's context.
- Link relevant docs from the Journal when a task produces notable documentation.

---

## Folder Structure

```
var_demo/
├── CLAUDE.md        # This file — working agreement
├── Journal.md       # Shared prompt & outcome log
├── docs/            # Project documentation
└── tests/           # All test code
```

---

## General Principles

- Prefer simple, focused changes. Avoid over-engineering.
- Do not add unrequested features, comments, or abstractions.
- Validate assumptions before making broad changes — read before editing.
- Flag security concerns immediately if encountered.
