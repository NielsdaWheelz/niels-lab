# The Second Study — presentation spec

Status: **v1.1 — all §13 questions resolved 2026-09-05; awaiting an explicit "go"**.
Branch: `agent/second-study`. No code changes yet. Standards: `docs/rules.md`, `docs/testing-standards.md`.
Decision context: The Second Study brief, five walkable plates, and the Page-and-Wall
comparison (Claude artifacts, 2026-09); client picks recorded 2026-09-05.

## 1. Goal

Reset the presentation of nielseriknandal.com around five client decisions. Content is
byte-sacred and untouched; machine surfaces keep exact parity. The failure being corrected
(diagnosed in the study's survey): the v1 design is totally coherent and almost never
sensuous — nothing is granted scale, the evidence apparatus is visually timid, every room
wears the same shell.

## 2. The five decisions (locked, client 2026-09-05)

| Decision | Pick | Meaning |
| --- | --- | --- |
| Ground | **paper** | Light becomes the canonical theme (the leaf); dark becomes the desk-at-night scene. |
| Proof | **margin** | Evidence stands in a reserved outer margin beside its claim, at claim height, always visible on wide viewports; folds inline (disclosure) on narrow ones. |
| Order | **even** | One body weight everywhere; order comes from inks, rules, italics. Settling type declined. |
| Clock | **hour-alive** | The index epigraph and which list stands open follow the real San Francisco hour band, server-side, zero JS. |
| Ledger | **material law** | `/log` only: rows dry with age; a struck failure fades faster than the lesson attached to it; lessons never fade. |

Also by the client's hand: the **scroll threshold is cut** (2026-09-05). No monumental
entrance; the site's force comes from scale, margins, the hour, and the corpus itself.

## 3. Non-goals

- No threshold or monument of any kind; no settling/graded type; no painted skies or
  generative grounds; no strata/find-number apparatus; no sound.
- No foul-papers source comments (deferred: requires the client's real draft archive —
  nothing fabricated enters an evidence-sacred site; revisit when the archive question is answered).
- No content changes: list entries, essays, projects, CV data, register — all byte-sacred.
  The two exceptions are §5.6 (colophon rewrite, client-signed) and §13 Q2 (hour-table
  curation of existing lines, client-approved).
- No Homunculus build; no framework, nav, or route changes; no new client components;
  no new dependencies.

## 4. Target behaviour (only what changes)

| Route | Change |
| --- | --- |
| `/` | Incipit unchanged in copy; above the table of lists, an hour epigraph: one corpus line, verbatim, chosen by the SF hour band. The open-by-default list is chosen by the same band (replaces "first list open"). Contents rows keep their `<details>` structure; counts and last-written dates move to the margin on wide viewports. |
| `/lists/[slug]` | The canon leaf: asymmetric computed margins, running head, entries with margin glosses (evidence beside claim). Narrow viewports fold each gloss into its existing disclosure. |
| `/log` | The material law (§5.4). Row structure and content unchanged. |
| `/writing`, `/projects`, `/cv`, `/now`, `/lab`, 404 | Retokened to the leaf; layout shells adopt the canon grid; no structural change. Print CV: tokens only, still one page. |
| `/colophon` | Rewritten (client-signed) to document the new design: canon formula, hour tables, ink-age constants, the confessed client components. |
| `/og` | Rethemed to the leaf. |
| Machine surfaces | Pipeline untouched; content parity preserved. Every list remains fully present in HTML regardless of hour (the hour changes only which `<details>` carries `open`). |

## 5. Architecture (what changes)

1. **Canon layout** — `globals.css` gains the leaf grid: a text column (68ch measure kept)
   plus a reserved outer margin, ratios computed and published in the colophon; a running
   head (site title left, section right) on every leaf. Below the canon breakpoint the
   margin folds and the book becomes a scroll.
2. **Margin apparatus** — evidence renders as an aside immediately after its claim in DOM
   order (curl reads claim-then-proof linearly — better structure than today's collapsed
   details). Wide: grid places the aside in the outer margin at claim height. Narrow: the
   aside renders inside the existing `<details>` disclosure. One gloss per entry (the
   schema already guarantees 0..1). Zero JS.
3. **Hour system** — `src/lib/hour.ts`: pure `band(date, tz) → HourBand` and
   `season(date)`; `src/content/hours.ts`: the typed table (§6) referencing corpus entries
   by `(slug, index)` so epigraphs are byte-identical by construction. `/` renders with
   `revalidate: 3600`. Build-time fallback band: the small hours.
4. **Ledger ink-age** — `src/lib/ledger/inkAge.ts`: pure `ageStep(rowDate, today) → 0..4`
   mapped to stepped tokens `--age-0..--age-4` (defined in `globals.css`, both themes,
   every step AA — stepped, not continuous, so tokens remain the only color source).
   Failure rows use `ageStep(date, today, FAILURE_FACTOR)`; their lessons render in full
   celadon at any age. `prefers-contrast: more` renders everything at `--age-0`.
5. **Motion** — the site-wide budget stays one concept: the existing 250ms ink-reveal on
   disclosure open (index lists; narrow-viewport glosses). Wide-viewport list pages have
   no disclosures and therefore no motion. `prefers-reduced-motion` honored as today.
6. **Colophon** — documents: canon ratios, hour tables, ink-age constants
   (`AGE_STEPS`, `FAILURE_FACTOR`), theme flip, and the unchanged client-component
   allowance (theme toggle, lab playground).
7. **The wit law** — wit lives only where chrome must already speak; the site never
   creates a surface to carry a line. The enumerated surfaces (exhaustive until amended
   in writing): the 404 (exists), the ledger's stale state ("deload week", exists), the
   print CV footer, the colophon sign-off, the llms.txt tail. Each line is a true
   sentence in the register (sentence case, no exclamation, precision is the humor),
   drafted in W5 and shipped only with the client's signature. Neutral filler is the
   fallback for any line the client declines.

## 6. Schemas

```ts
// src/content/hours.ts
type HourBand = 'small-hours' | 'dawn' | 'morning' | 'afternoon' | 'dusk' | 'evening'
// bands (America/Los_Angeles): 00–05 · 05–08 · 08–12 · 12–17 · 17–20 · 20–24

type HourTable = Record<HourBand, {
  epigraph: { list: string; index: number } // corpus entry, rendered verbatim
  opens: string                             // published list slug, default-open on /
}>

// src/lib/ledger/inkAge.ts
const AGE_STEPS = [0, 90, 180, 365, 730]    // days → --age-0 .. --age-4
const FAILURE_FACTOR = 2                    // a failure dries twice as fast as its cohort
// lessons are exempt: full celadon at any age
```

`validate-content.mjs` grows: every `HourTable` ref resolves to a published list and an
in-range entry; all six bands present; `opens` slugs published; ink-age constants exported
and finite. A table that cannot prove itself fails the build.

## 7. Design tokens

The canonical theme flips: **light is the leaf, dark is the desk at night.** Roles and
faces are unchanged (Newsreader all reading text, Quattro chrome; 68ch; oldstyle nums;
hanging em-dashes; nothing uppercase-tracked; no third face).

| Token | Leaf (default) | Night | Role |
| --- | --- | --- | --- |
| `--ink` | `#F5F3F7` → §13 Q1 | `#16161D` | page ground |
| `--lamplight` | `#16161D` | `#F1EDF5` | body text |
| `--murasaki` | `#66327C` | `#A87BC4` | headings, links |
| `--celadon` | `#4A7367` | `#7FA99B` | evidence glosses, dates, counts, lessons |
| `--rule` | `#DDD6E4` | `#2A2833` | hairlines, running head rule |
| `--faded` | `#6E6878` | `#8F8A99` | secondary text; ink-age floor |
| `--age-0..4` | interpolated `--lamplight`→`--faded` | same, night values | ledger ink-age steps, each AA |

Night is a scene, not an inversion: on wide viewports the dark theme renders the canon
text block as a faintly lighter field on the deep ground — the lamplit leaf on the desk —
from two flat tones and geometry, no gradients.

## 8. Hard-cutover kill list

- Wide-viewport evidence `<details>` (superseded by the margin aside; narrow keeps the fold).
- Dark-as-default in `theme.ts` (default flips; toggle and storage reused as-is).
- "First list open" on `/` (superseded by the hour rule).
- The v1 `globals.css` visual layer — replaced, not amended, per house rule.

## 9. Reuse

`site.ts`, `StructuredData`, `content.ts`, MDX renderer, ledger pipeline
(`LedgerSource`, GitHub ISR), `theme.ts` engine, `cv/print.css` pattern,
`validate-content.mjs` (extended), both fonts.

## 10. Workstreams (non-overlapping)

| # | Scope | Files owned | Depends on |
| --- | --- | --- | --- |
| W1 | Tokens, canon grid, running head, theme flip, OG | `globals.css`, `layout.tsx`, `components/{nav,footer,ThemeToggle}`, `og/route.tsx` | — |
| W2 | Index + list leaves + margin apparatus | `app/page.tsx`, `app/lists/`, gloss component + module css | W1 |
| W3 | Ledger material law | `lib/ledger/inkAge.ts` (+test), `app/log/` | W1 |
| W4 | Hour system | `lib/hour.ts` (+test), `content/hours.ts`, `/` integration | W1, §13 Q2 |
| W5 | Inner restyle sweep + colophon draft + §5.7 chrome lines (for signature) | `writing/`, `projects/`, `cv/`, `now/`, `lab/`, `not-found.tsx`, `colophon/` | W1 |
| W6 | Lint + machine surfaces | `scripts/validate-content.mjs`, `README.md` | W2–W4 schemas |
| W7 | QA: `bun run check`, screenshots both themes at all six bands, slop audit, curl parity | none (read-only) | all |

W2/W3/W4/W5 run in parallel after W1. W4's table lands only after the client approves it (§13 Q2).

## 11. Acceptance criteria

- `bun run check` green; `hour.ts` and `inkAge.ts` fully covered by colocated tests
  (fixed dates, no clock — per testing standards).
- `curl` of every route: all load-bearing text present; every evidence-bearing entry reads
  claim-then-proof in linear DOM order.
- Every token pair AA in both themes, **including all five age steps**; `prefers-contrast:
  more` yields full ink; `prefers-reduced-motion` honored.
- Zero new client JS; the two confessed client components remain the only ones.
- The six bands render the approved epigraph and open list (unit-tested at boundary
  minutes); machine parity: all eight lists complete in HTML at every hour.
- In `/log`, a failure row renders visibly drier than same-age neighbors and its lesson
  renders full celadon; validated constants match the colophon's published values.
- Wit only on the §5.7 enumerated surfaces; every shipped chrome line client-signed; no
  surface exists whose only purpose is to carry a line.
- Print CV one page, black-and-white legible; OG in leaf style; bio byte-identical everywhere.
- Slop audit passes under `docs/rules.md` as amended by the §13 Q1 resolution.

## 12. Key decisions

| Decision | Rationale |
| --- | --- |
| Light canonical | Client pick ("paper"), 2026-09-05. Night survives as a true scene, not an afterthought. |
| Margin aside over wide-screen `<details>` | The site's ethic made spatial; DOM order improves for curl and crawlers; narrow fold keeps the zero-JS degradation story. |
| Even order | Client pick; settling was the wall's spine and is not grafted piecemeal. |
| Ink-age as stepped tokens | Continuous interpolation would put computed hex outside `globals.css`; steps keep the token rule and make AA checkable per step. |
| Hour via ISR 3600 | Static-first honored; no client clock; band logic is a pure tested function. |
| Index keeps its `<details>` table | Hour-alive needs an openable list on `/`; the v1 structure already provides it — least churn, no new pattern. |
| Wit law: chrome speaks in register only where it must speak at all | Client pick (option c, 2026-09-05); generalizes "deload week"; caps surface area structurally, not by willpower. |
| Threshold cut | Client, 2026-09-05. Recorded; not relitigated. |

## 13. Questions

1. **Leaf warmth and the rubric — RESOLVED (client, 2026-09-05):** the cool leaf
   `#F5F3F7`, no fourth ink, zero deviation from `rules.md`. The warm leaf and vermilion
   rubric of the study's demos do not ship.
2. **The hour table — RESOLVED (client, 2026-09-05):** mechanism approved. The six
   epigraph pairings (corpus lines, verbatim, referenced by index) and six open-list
   assignments still come to the client for sign-off before W4 lands — that gate stands.
3. **Wit — RESOLVED (client, 2026-09-05): option (c),** codified as §5.7. Wit is
   confined to surfaces where chrome must already speak; no surface is ever created to
   carry a line; every line ships only with the client's signature.
