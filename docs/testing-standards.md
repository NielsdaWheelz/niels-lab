# Testing standards

Codified 2026-08-09 alongside `docs/rules.md`. Red/green/refactor is the loop; this file
defines what gets tested, with what, and what survives.

## Runner & layout

- `bun test` (Bun's built-in runner; zero test dependencies). Tests are colocated:
  `src/lib/foo.ts` → `src/lib/foo.test.ts`.
- `bun run test` runs in the `check` chain before `build`. A red test blocks commit.

## What gets tested

- Pure logic in `src/lib/`: parsing, merging, sorting, mapping, validation, formatting.
  Every exported function with branching behavior gets tests for its target behavior
  before implementation (red first).
- Content invariants that types cannot express (date formats, slug uniqueness, evidence
  URL shape) — enforced in `scripts/validate-content.mjs`, which gates the build.

## What does not get tested

- React rendering, RSC pages, CSS, and markup. The page-level gates are `next build`,
  `validate-content.mjs`, and zero-JS parity (`curl` contains every load-bearing sentence).
  No snapshot tests, no DOM emulation, no mocking frameworks.

## The loop

1. Red: write the failing test that names the target behavior from the spec.
2. Green: minimal code to pass.
3. Refactor: under green, until the code meets `docs/rules.md`.
4. Prune: delete scaffolding tests — any test that pins implementation detail or is
   subsumed by a stronger assertion. A test earns its place by failing when observable
   behavior regresses, and only then.

## Determinism

- No network, no clock, no filesystem in tests, except explicit inline fixtures.
- External APIs (e.g. GitHub events) are tested through their pure mapping functions
  against captured sample payloads pasted into the test file.
