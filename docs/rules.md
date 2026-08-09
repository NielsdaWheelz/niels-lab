# Codebase rules

Codified 2026-08-09 from observed conventions plus Pillow Book spec decisions. These are the
standards; deviations need a written reason in the PR.

## Stack & tooling

- Bun-first: `bun install`, `bun run <script>`. Node pinned via `engines`.
- `bun run check` (format:check → lint → typecheck → build) must pass before any commit.
- Content validation (`scripts/validate-content.mjs`) gates the build; extend it when a
  content schema grows — never bypass it.
- Dependencies are justified individually. Prefer the platform (semantic HTML, `<details>`,
  CSS) over a package. Removing a dep is a feature.

## TypeScript

- Strict mode; no `any`, no non-null assertions without a comment stating the invariant.
- Structured content is typed TS data — type-checking is the CMS. Prose is MDX with
  `---` frontmatter, one `key: value` per line, dates `YYYY-MM-DD`.
- Shared logic lives in `src/lib/`; route-specific code stays colocated with its route.
- One source of truth per concern; duplicated parsing/loading logic is a bug (see the
  former `writing/utils.ts` / `projects/utils.ts` twins).

## Rendering

- RSC-first. Client components only for irreducible interactivity; each one is listed in
  the colophon. Current allowance: the theme toggle and the lab's sampling playground
  (kept per spec §3; spec §11's "theme toggle only" line predates that decision).
- Zero-JS content parity: every load-bearing fact and sentence must exist in the initial
  HTML response. If `curl` can't see it, it doesn't exist.
- Static-first; ISR (`revalidate`) for external data. No client-side fetching for content.

## CSS & design

- Hand-rolled CSS only: one `globals.css` of tokens and primitives, CSS modules per
  component. No Tailwind, no UI kits, no CSS-in-JS.
- Design tokens are the only source of color and type; hex literals outside `globals.css`
  are forbidden.
- Both themes always; every token pair passes WCAG AA in both.
- Motion is scarce and purposeful. Site-wide budget: one animation concept, honored under
  `prefers-reduced-motion` with a still equivalent.
- Banned tells (first- and second-order AI-slop): Inter/Geist-by-default, uppercase-tracked
  eyebrow labels, numbered section indexes, card grids, gradients, glassmorphism,
  fade-up-stagger entrances, italic accent word in headlines, cream/terracotta palettes,
  sparkle icons, "built with ♥" footers.

## Copy

- Sentence case. No exclamation marks. No marketing verbs (unleash, streamline, supercharge).
- Claims are either backed by a link to primary evidence or written as plain observation —
  never decorated with fake precision.
- The register is the author's: precise, dry, concrete. If a sentence could appear on any
  engineer's site, rewrite or delete it.

## Accessibility

- Semantic HTML first; skip link; visible `:focus-visible` on every interactive element.
- Keyboard path for every interaction; `<details>`/`<summary>` preferred for disclosure.

## Hygiene

- No dead code, no commented-out blocks, no legacy fallbacks — delete; git history is the
  archive. Hard cutovers only.
- Comments state invariants and constraints, not narration.
- Branch → PR to `main`. Commit subjects lowercase and descriptive; the why goes in the body.
- Machine surfaces (`robots`, `sitemap`, `rss`, `llms.txt`, JSON-LD) update in the same PR
  as the content they describe; the bio string is byte-identical everywhere it appears.
