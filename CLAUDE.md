# CLAUDE.md — Working Agreement for var_demo

## Journal
- Maintain `Journal.md` at the project root.
- Append every prompt and its outcome after each interaction.
- Include the date, a sequential entry number (001, 002, …), and the time taken.

## Planning
- Before starting any **major** task, propose a written plan and wait for approval.
- Minor tasks (small edits, quick lookups, trivial fixes) may proceed without a plan.
- "Major" means: new features, architectural changes, multi-file refactors, or anything with meaningful risk.

## Test-Driven Development
- All test code lives in the `tests/` folder.
- For every task where a test is feasible, write the test alongside (or before) the implementation.
- Major tasks begin with a **test plan** listing what will be tested and why.
- Record each test and its outcome in `Journal.md`.

## Documentation
- All documentation goes in the `docs/` folder.
- Keep docs up to date as features are added or changed.

## Version Control
- A `.gitignore` is maintained at the project root (Python + macOS rules already in place).

## General Conventions
- Always read a file before editing it.
- No unrequested features, comments, or abstractions — keep changes minimal and focused.
- Use clear, sequential Journal entry numbers (001, 002, …).
