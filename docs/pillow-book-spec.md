# The Pillow Book — redesign spec

Status: **v1.0 — open questions resolved 2026-08-09; ready for implementation on "go"**.
Branch: `agent/pillow-book`. No code changes yet. Codebase standards: `docs/rules.md` (codified as part of this work).
Decision context: full design research + four concept plates in the design brief (Claude artifact
"The Redesign — Four Directions", 2026-08). Client chose **The Pillow Book** with the
**Homunculus recorded as v2**; job = peer reputation; live data = yes; register = austere, one delight.

## 1. Goal

Rebuild nielseriknandal.com as a zuihitsu: the site IS a set of Shōnagon-style lists in which
every checkable claim is wired to primary evidence. One register holds all three selves
(athlete, reader, engineer). Kill every catalogued AI-slop tell, first- and second-order.

## 2. Non-goals

- No Homunculus build (v2; plan recorded in `/colophon`, nothing else).
- No new essay/project content beyond frontmatter/positioning fixes.
- No CMS, no database, no auth, no comments, no analytics changes.
- No Tailwind adoption; no component library; no framework migration.
- No external ledger adapters beyond GitHub in v1 (contract defined, adapters later).
- No visual redesign of the print CV beyond tokens (it is the recruiter escape hatch).

## 3. Target behaviour

| Route | State | Behaviour |
| --- | --- | --- |
| `/` | rebuild | The book's index: title line "The Pillow Book of Niels-Erik Nandal, written in San Francisco.", table of lists (title, entry count, last-written date), first list open by default, "Things I have built" pinned second. Quiet footer routes to /cv and email. No hero, no photo, no CTA band. |
| `/lists/[slug]` | new | One list per page: all entries, hanging em-dash markers, evidence disclosures inline. Shareable permalink; in sitemap, RSS, llms.txt. |
| `/log` | new | The living ledger: unified dated rows (training / reading / shipping), failures kept in with strikethrough + one-sentence lesson. GitHub commit rows auto-refresh (ISR, daily). Honest stale state: after 14 quiet days a single line "deload week." |
| `/writing`, `/writing/[slug]` | restyle | Essays. Book typography; content unchanged. |
| `/projects`, `/projects/[slug]` | restyle | Working notes per project; content unchanged. Targets of "Things I have built" evidence links. |
| `/cv` | restyle + data fix | Plain, dense, print-ready. Add Solid entry; drop "open to roles" framing. |
| `/now` | restyle + rewrite | Hand-tended; becomes a dated diary page in book register. |
| `/colophon` | rewrite | Documents new design (tokens, faces, decisions), provenance, and the Homunculus v2 plan ("forthcoming" — the Pessoan move). |
| `/lab`, `/lab/sampling` | reshell | Keep sampling playground; restyle shell only. |
| `not-found` | rebuild | A tiny list: "Things that are not where you left them." This is the site's one delight. |
| `/og` | retheme | Book-styled OG card (ink ground, Newsreader, murasaki). |
| `/rss`, `/sitemap`, `/robots`, `/llms.txt`, `/llms-full.txt` | regenerate | Include lists + log. Content parity with HTML. |
| ctrl+k terminal, assistant panel | **kill** | Hard cut. No fallback, no flag. |

## 4. Architecture

Next.js 16 App Router, RSC-first, static-first. Four subsystems:

1. **Content** — `src/lib/content.ts`: ONE frontmatter/MDX loader (consolidates the duplicated
   `writing/utils.ts` + `projects/utils.ts`) plus typed list data in `src/content/lists.ts`.
   Lists are code-reviewed TypeScript data, not MDX: entries are one-liners with structured
   evidence; type-checking is the CMS.
2. **Evidence** — pure markup: each evidence-bearing entry renders a `<details>` disclosure
   (celadon mark as `<summary>`), containing label + link (or embedded asset). Zero JS;
   works in reader mode, curl, and AI crawlers. Entries without proof render with NO mark —
   a dead or empty evidence link is a build error (validated).
3. **Ledger** — `src/lib/ledger/`: `LedgerSource` interface; v1 sources: `manual.ts`
   (reads `src/content/log.ts`) and `github.ts` (public events API, server-side fetch,
   `revalidate: 86400`, no token). Merge + sort in `getLedger()`. Rendering is RSC; zero client JS.
4. **Design system** — rewritten `globals.css` (tokens below) + `next/font`:
   Newsreader (already wired via `next/font/google`; keep, normal+italic, `font-optical-sizing: auto`)
   and iA Writer Quattro (vendor woff2 + OFL license, `next/font/local`) for dates/counts/chrome.
   Geist and Caveat removed.

Machine surfaces: `site.ts` stays the single metadata factory; `StructuredData` upgraded
(Person JSON-LD, deep `sameAs`, byte-identical bio string shared with visible About text);
`validate-content.mjs` extended to validate list schemas + evidence URLs.

## 5. Schemas

```ts
// src/content/lists.ts
type Evidence =
  | { label: string; href: string }            // primary proof: commit, video, meet result, post
  | { label: string; src: string; kind: 'image' } // embedded asset in /public/evidence/

type Entry = {
  text: string          // 5–25 words, one image or one judgment
  evidence?: Evidence   // omit when no proof exists; never placeholder
  added: `${number}-${number}-${number}` // YYYY-MM-DD
}

type PillowList = {
  slug: string          // kebab-case, stable, used in URL + anchors
  title: string         // the merciless heading
  note?: string         // optional one-line gloss under the heading
  entries: Entry[]      // ordered by the author, not by date
}
// Derived (never stored): count = entries.length; lastWritten = max(added).

// src/lib/ledger/types.ts
type LedgerEvent = {
  date: string                     // YYYY-MM-DD
  kind: 'train' | 'read' | 'ship' | 'write'
  text: string                     // logbook shorthand: "SQ 5×5 @365 · Braudel 41pp"
  failed?: { lesson: string }      // renders struck-through + lesson
  href?: string
}
interface LedgerSource { get(): Promise<LedgerEvent[]> }
```

## 6. Design tokens

| Token | Dark (default) | Light | Role |
| --- | --- | --- | --- |
| `--ink` | `#16161D` | `#F5F3F7` | page ground |
| `--lamplight` | `#F1EDF5` | `#16161D` | body text |
| `--murasaki` | `#A87BC4` | `#66327C` | headings, links (Heian purple; AA on both grounds) |
| `--celadon` | `#7FA99B` | `#4A7367` | evidence marks, dates, counts |
| `--rule` | `#2A2833` | `#DDD6E4` | hairlines |
| `--faded` | `#8F8A99` | `#6E6878` | secondary text |

Type: Newsreader for all reading text (index headings italic display; 68ch measure; `oldstyle-nums`;
book-style paragraph indents in prose: `p{margin:0} p+p{text-indent:1.5em}`). Quattro for dates,
counts, evidence chrome, ledger rows. NOTHING is uppercase-tracked; no third face.
Motion: exactly one animation — a 250ms `clip-path` ink-reveal when a list opens, once,
disabled under `prefers-reduced-motion`. Nothing else moves. Entry markers are hanging
em-dashes (`text-indent` negative), not bullets.

Dark is the canonical theme ("night writing"); light is a true inverse, both AA. Existing
`theme.ts` engine + toggle reused as-is (restyled).

## 7. Hard-cutover kill list

- Components: `Terminal/*`, `TerminalLauncher`, `SketchCanvas`, `SystemMap`, `NeuralPathway`,
  `NeuralLoading`, `DrawHeading`, `ContentReveal`, `TextReveal`, `NotFoundScene`.
- Lib: `filesystem.ts`, `terminal.ts`, `nielsGpt.ts`, `sse.ts` (niels-gpt chat is superseded;
  Homunculus v2 reintroduces the model properly).
- Deps: `geist`, `roughjs`; `tailwindcss` + `@tailwindcss/postcss` (verify unused → remove);
  Caveat font.
- CSS: graph-paper ruling, film grain, custom cursors, ink stamp, circle annotation,
  marginalia hand-font, fade-up staggers, system-map/proof-strip/principles-grid styles —
  the entire current `globals.css` is replaced, not amended.
- Copy: "Build the magic. Keep the receipts.", kickers, section indexes, "open to roles".
- Env: `NEXT_PUBLIC_NIELS_GPT_API_BASE`.

## 8. Reuse / consolidate

- `site.ts` metadata factory, canonical-URL + robots logic — reuse untouched.
- `theme.ts` + init script — reuse.
- MDX renderer (`components/mdx.tsx`) + `sugar-high` — reuse; retokenize highlight colors.
- `writing/utils.ts` ≡ `projects/utils.ts` → one `src/lib/content.ts` (duplication extirpated).
- `validate-content.mjs` — extend (lists schema, evidence URL liveness, date formats).
- `cv/print.css` pattern — reuse.
- ~~`public/projects/*.svg` art — reuse in project pages~~ (amended during QA 2026-08-09: the
  premise was wrong — the SVGs were cream/terracotta-palette cards, the banned tell itself;
  extirpated with their frontmatter and validator branch).

## 9. Content design contracts

Each list has a designated content designer (writer agent) working to this contract; the
curator (lead) cuts to the bar. **The bar: an entry survives only if no other engineer's site
could plausibly carry it.**

- Form: one image or one judgment, 5–25 words. No exclamation marks, no "delve/tapestry",
  no LinkedIn energy, no twee. Precision is the humor.
- Register mix is mandatory per list: iron/ice/grass + page + terminal must interleave;
  an all-gym or all-code list fails review.
- Evidence: ≥ one-third of entries site-wide carry live proof (repo, commit, HF, post).
  Claims without available proof ship unmarked — never a placeholder.
- **Lifting/meet claims are prose, never evidence-marked** (client decision 2026-08-09):
  the numbers appear as written observations only; no lifting entry enters the evidence system.
- Launch corpus: 8 lists × 6–10 entries (~60 total, curated from ~120 candidates).
- Launch lists: Things that quicken the heart · Things I have built · Hateful things
  (found in codebases) · Elegant things · Things that should be small · Rare things ·
  Things that are distant though near · Things now in decline. (Bench: near-though-distant,
  seen-at-dawn, squalid, awkward, gains-by-repetition — publish later as diary cadence.)
- "Things I have built" doubles as the résumé: each entry links its `/projects/` page or repo;
  written in list register, not summary register.
- Essays/projects MDX: content untouched (voice already right); only frontmatter/links audited.

## 10. Workstreams (non-overlapping)

| # | Scope | Files owned | Depends on |
| --- | --- | --- | --- |
| W0 | Codebase standards | `docs/rules.md` | — (done pre-implementation) |
| W1 | Kill list + design system + fonts + layout shell + OG | `globals.css`, `layout.tsx`, `components/{nav,footer,ThemeToggle}`, `og/route.tsx`, `package.json`, deleted files | — |
| W2 | Content model + homepage + lists + 404 | `lib/content.ts`, `content/lists.ts`, `app/page.tsx`, `app/lists/`, `not-found.tsx` | W1 tokens |
| W3 | Inner pages restyle | `writing/`, `projects/`, `cv/`, `now/`, `colophon/`, `lab/` (page shells + module css) | W1 tokens |
| W4 | Ledger | `lib/ledger/`, `content/log.ts`, `app/log/` | W1 tokens |
| W5 | Machine surfaces | `site.ts`, `StructuredData.tsx`, `llms*/route.ts`, `rss/route.ts`, `sitemap.ts`, `scripts/validate-content.mjs`, `README.md` | W2 schemas |
| W6 | Content corpus | writer/curator output → `content/lists.ts` data values only | §9 contract |
| W7 | QA: `bun run check`, screenshots both themes, slop-tell lint, evidence-link liveness | none (read-only) | all |

W2/W3/W4 touch disjoint files and run in parallel after W1. W6 lands as data into W2's file
(coordinated single merge). CV data fix (`cv/data.ts`) belongs to W3.

## 11. Acceptance criteria

- `bun run check` green (format, lint, typecheck, build).
- Zero references to killed components/deps; `git grep -iE 'geist|roughjs|terminal|caveat'` clean in `src/`.
- Client JS ≤ ~30 KB beyond framework + analytics; the theme toggle and the lab's sampling
  playground (§3 keeps it) are the only client components (amended during QA 2026-08-09 —
  the original "toggle only" line contradicted §3).
- `curl` of every route contains all load-bearing text (zero-JS parity, AI-crawler ready).
- All token pairs pass WCAG AA; both themes shipped; `prefers-reduced-motion` honored.
- Slop-tell audit passes: no Inter/Geist, no uppercase-tracked labels, no numbered section
  indexes, no card grids, no gradient, no fade-up staggers, nothing cream/terracotta.
- Corpus: ≥ 8 lists, ≥ 50 curated entries, every evidence `href` resolves (validated in CI).
- Positioning: senior engineer at Solid on `/`, `/cv`, `/now`, JSON-LD, llms.txt — byte-consistent bio.
- OG card renders in book style; RSS + llms.txt include lists and log.
- Homunculus v2 plan recorded in `/colophon`.
- Print CV still one page, still legible in black and white.

## 12. Key decisions

| Decision | Rationale |
| --- | --- |
| Lists as typed TS data, not MDX | entries are one-liners with structure; type-checking is the CMS; MDX adds parse surface for nothing |
| `<details>` for evidence + list expansion | zero JS, no-JS parity, reader-mode safe; the anti-Bruno-Simon move |
| Terminal + niels-gpt chat cut | off-concept, heaviest JS, aesthetic of the old site; Homunculus v2 is its rightful successor |
| Dark canonical, light inverse | "night writing" is the concept; light is a real theme, not an afterthought |
| Newsreader stays | already in the stack, has real opsz/italics; the change is role (primary, not garnish) |
| Hand-rolled CSS stays | matches repo standard; Tailwind dep removed if grep-confirmed unused |
| GitHub-only live source in v1 | no credentials, no promise-to-break; adapters land behind `LedgerSource` |
| No push until client approves | branch `agent/pillow-book`; PR after review |

## 13. Resolved questions (client, 2026-08-09)

1. `docs/rules` did not exist → **codified as `docs/rules.md`** in this work; implementation follows it.
2. Solid entry: **Senior Software Engineer, Solid · March 2026 – present** (CV, JSON-LD, bio — byte-consistent).
3. Lifting claims: **excluded from the evidence system**; numbers live in prose only (see §9).
4. Ledger v1: **manual entries in `content/log.ts`** + GitHub public API; adapters later behind `LedgerSource`.
