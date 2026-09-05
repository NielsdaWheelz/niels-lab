# niels-lab — design survey for redesign

## 1. Current design language

**Metaphor:** the site *is* a book — "The Pillow Book of Niels-Erik Nandal, written in San Francisco," a zuihitsu after Sei Shōnagon. Dark is canonical ("night writing"); light is a true inverse, not an afterthought.

**Tokens** (globals.css, the only source of color; hex elsewhere is forbidden): `--ink` #16161D ground · `--lamplight` #F1EDF5 body · `--murasaki` #A87BC4/#66327C headings+links (Heian purple) · `--celadon` #7FA99B/#4A7367 evidence marks, dates, counts · `--rule` hairlines · `--faded` secondary. All pairs computed AA in both themes.

**Faces:** Newsreader (variable opsz, italic display for titles) for all reading text; iA Writer Quattro (vendored woff2, OFL) for dates/counts/chrome only. No third face; nothing uppercase-tracked.

**Spacing philosophy:** one 44rem column, 68ch measure, book paragraph indents (`p+p{text-indent:1.5em}`), hanging em-dash entry markers (negative text-indent, never bullets), oldstyle nums in prose / tabular in chrome. Motion budget: exactly one animation, a 250ms clip-path ink-reveal when a `<details>` opens; disclosure marks are `›` that rotate with an instant swap.

**Where it succeeds:** total coherence — six tokens, two faces, one animation, zero clutter; the anti-slop lint (no gradients, no card grids, no Inter, no eyebrow labels) is fully honored; the 404 is genuinely delightful. **Where it is merely austere:** the homepage is a wall of identical closed disclosure rows; every page is the same centered column with no rhythm shifts; murasaki and celadon are used at small sizes only — the palette never gets a full-bleed moment; the OG card is the only place ink+Newsreader are composed at display scale; the print CV, /lab, /log all wear the same undifferentiated shell. Nothing is ugly; almost nothing is sensuous.

## 2. Content inventory

**Routes:** `/` (book index: title line, two sublines, table of 8 lists as `<details>`, first open) · `/lists/[slug]` (permalink per list) · `/log` (ledger: dated rows train/read/ship/write, failures struck-through with lesson, GitHub rows via ISR daily, "deload week" stale state) · `/writing` + 26 MDX posts (fractal-week 1–12, solid-week/month diaries, solid-interview/negotiation, zero-to-hero 1–2, my-first-project) · `/projects` + 15 MDX writeups (niels-gpt-1/2/app, nexus, factory-simulator, fractal-chat, fractal-go, suno-demo, hyperion-era entries like ariel, agency, llm-tools…) · `/cv` (dense, print.css, JSON-LD ProfilePage) · `/now` · `/colophon` · `/lab` + `/lab/sampling` (client-side sampling playground) · `not-found` · machine surfaces: `/og`, `/rss`, `/sitemap`, `/robots`, `/llms.txt`, `/llms-full.txt`.

**The 8 published lists** (61 entries; one draft list "read", 11 entries, hidden): Things that quicken the heart (7) · Things I have built (9, "The résumé. Every line links to the thing itself.") · Hateful things (found in codebases) (8) · Elegant things (8) · Things that should be small (8) · Rare things (7) · Things that are distant though near (7) · Things now in decline (7).

**Facets and where they live:** powerlifting/strongman (quicken-the-heart, elegant, should-be-small, distant-though-near, log rows "SQ 3×3 @385"); hockey (saucer pass, shootout net, bag skate, the 404's puck); soccer (keeper off his line, "Eleven wingers a side now"); reader/novelist/poet (Levin mowing, Dickinson's slant rhyme, Braudel, Ulysses, two Keats odes kept in memory, the draft reading list); AI researcher (RMSNorm, weight tying, tokenizer round-trips, model cards, the forthcoming Homunculus); behavioural-economics past (p = 0.051, preregistration, incentivized choice experiments); engineer everywhere else.

**Verbatim flavor:**
- "Squat bar path: one vertical line, drawn twice."
- "Weight tying: the matrix that reads is the matrix that speaks."
- "Braudel: the sea gets the biography, and kings are demoted to weather."
- "Sleep, with the diff still open."
- "The end of Ulysses, from anywhere in Ulysses."
- "A mock of the unit under test. The suite is green because it agrees with itself."

## 3. The Homunculus v2 plan

Recorded only as one colophon paragraph ("Homunculus, forthcoming") — the deliberate Pessoan move; the spec's non-goals forbid building it in v1. Verbatim: "Version two of this site will contain the Homunculus: a small language model trained on my own writing, answering questions as the site rather than about it. Pessoa kept heteronyms; I intend to keep one made of matrices. It is not built, and this paragraph is currently its entire implementation."

## 4. Tech substrate

Next.js 16 App Router, RSC-first, static-first, Bun tooling; `bun run check` gates commits. Hand-rolled CSS only (one globals.css + CSS modules; no Tailwind, no kits). Lists are typed TS data in `src/content/lists.ts` — "type-checking is the CMS"; `validate-content.mjs` fails the build on dead evidence links. **Zero-JS evidence pattern:** evidence is a plain `<details class="evidence">` with celadon Quattro summary — works in curl, reader mode, AI crawlers; the rule is "If `curl` can't see it, it doesn't exist." Only two client components allowed (theme toggle, lab sampling playground); each is confessed in the colophon. Machine surfaces: `site.ts` single metadata factory with byte-identical canonical bio across JSON-LD/llms.txt/homepage; OG route themed in book style; RSS/sitemap/robots/llms include lists+log. **Must not break:** zero-JS parity, evidence validation, both-themes AA, the register (docs/register.md governs every word — byte-sacred corpus), print CV one page, machine-surface parity. **Free to discard:** every visual decision — the spec itself says globals.css is "replaced, not amended" when a design changes; layout, tokens, and the one-animation budget are design choices, not substrate (though rules.md's slop-tell ban and motion-scarcity principle stand).

## 5. Raw material

1. **The corpus is a poem the design sets like a memo.** Lines like "Sleep, with the diff still open" are given identical 1.125rem body treatment as furniture text — nothing is ever allowed display scale, silence, or a page of its own.
2. **Murasaki and celadon are a genuinely Heian palette used only as trim** — no ink-wash ground, no large field of purple, no material texture of paper/lamp/night despite "night writing" being the stated concept.
3. **The evidence marks are the site's moral signature** — claim wired to proof — yet render as a 13px `›`; the most distinctive idea on the site is visually its most timid.
4. **The log's strikethrough failures with attached lessons** ("A build that passes locally proves the laptop, not the deploy.") are a diary device with real pathos, currently indistinguishable rows in a plain `<ol>`.
5. **The interleaved registers — barbell beside Braudel beside RMSNorm within a single list** — are the whole persona, but the typography never marks the turn; a design that let iron, page, and terminal feel materially different per entry (without breaking one-register prose) has an untouched seam to work.
