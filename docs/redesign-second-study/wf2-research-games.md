# Games as Art — Stealable Mechanisms for a Zuihitsu Site

## WORKS

- **Kentucky Route Zero** — theatrical staging: flat, layered set-pieces and stage directions instead of scenery: substance from *implication*, distance kept deliberately proscenium.
- **Outer Wilds** — progression *is* knowledge: no upgrades exist; the game never changes, only what you understand does. The avatar never grows — the player does.
- **Her Story** — search as authorship: the database is fixed, but the *order of discovery* is yours; you build the narrative by choosing what to query.
- **Disco Elysium** — the self as a quorum: skills are not stats but voices that interrupt with argument and dissent, externalizing a fractured mind as dialogue.
- **Journey / Ico** — wordless intimacy: meaning built entirely from gesture, proximity, and withheld explanation; restraint is the content.
- **A Dark Room** — interface that blooms: begins as one button; complexity is *earned*, never dumped, and the sparse UI is never decorative filler.
- **Frog Fractions** — genre betrayal: builds a boring, legible expectation, then detonates it — the surprise only works because the setup was played completely straight.
- **Universal Paperclips** — mechanics as argument: starts as a clicker, ends as an essay on instrumental reasoning; theme is never stated, it is *executed*.
- **Device 6** — typography as geography: text direction, layout, and shape encode movement and space; the page itself becomes the map.
- **notpron / ARGs** — the raw material is the puzzle: metadata, filenames, source code, and page structure all carry meaning for those who actually look.

## STEALABLE MOVES

1. **Knowledge, not points, as the only progression.** (Outer Wilds) The site should never gamify itself with badges or completion meters — the *only* thing that changes across a visit is what the reader now understands. A returning reader who has read the /cv should find that an entry in the zuihitsu lists now lands differently, because context — not unlocked content — has accumulated in their head. Structure cross-references so later sections retroactively re-arm earlier ones.

2. **Committee-of-voices marginalia.** (Disco Elysium) This is the single most resonant transplant given six facets in one man. Let a list entry occasionally carry a second, competing voice in the disclosure — the strongman's gloss on the poet's line, the engineer's dry footnote on the novelist's claim. Not a gimmick UI, just typographically distinct marginal notes that dramatize the internal argument between facets, rather than smoothing them into one narrator.

3. **Search as the narrative engine, not just retrieval.** (Her Story) The /log ledger shouldn't only scroll chronologically. A reader who filters by "training" versus "shipping" versus "reading" should feel they are *assembling* a portrait of the man, not browsing a table. This can be done zero-JS via native `<details>`/checkbox-hack faceting — the mechanism (reader chooses the query, gets a partial cut of the truth) survives without a framework.

4. **Theatrical staging over page templates.** (Kentucky Route Zero) Treat sections as acts and entr'actes, not URLs. A short, stage-direction-register line between major sections ("Enter the ledger.") does more work than a nav breadcrumb — it signals the whole site is one composed piece, not eight interchangeable CMS templates.

5. **Bloom, don't dump.** (A Dark Room) The homepage should open nearly bare — austere, almost withholding — and let the site's actual density (the dense CV, the sixty-line lists) arrive only as the reader goes looking for it. Complexity revealed is dignity; complexity dumped up front is a resume screaming for attention.

6. **Typography as spatial argument.** (Device 6) Let form encode content directly: the CV block should typographically *feel* dense and load-bearing (tight leading, no ornament); a poem entry should get room to breathe that a ledger entry doesn't. Don't caption the difference — let the reader's eye register it as meaning, the way Device 6's scrolling direction *is* the character's movement.

7. **Reward the one who reads the source.** (notpron, curl-clean HTML) The client's own machine-legibility requirement is a gift here: hide something for the reader who views source, follows a `<!-- comment -->`, or actually fetches the HTML with curl instead of a browser — a wry aside in a meta tag, an honest colophon note in the raw markup. This flatters the exact audience (engineers, researchers, agents) already implied by the site's SEO/curl-clean mandate, at zero cost to the polished surface.

8. **Withhold the explanation.** (Journey / Ico) Resist the urge to gloss every zero-JS disclosure with "click to expand" language or hover hints explaining what will happen. Let the disclosure triangle itself, unlabeled, be the entire invitation — trust the reader the way Journey trusts a wordless chime.

9. **One structural betrayal, never more.** (Frog Fractions) Somewhere — maybe /now, maybe /colophon — the site can play something completely straight for its whole length and then, once, refuse to be what its category promised (a "now page" that is actually a single confession; a colophon that turns out to be the most personal essay on the site). This works only in a single, well-chosen location, and only because everything else on the site kept its genre promise scrupulously.

## WHERE GAME BECOMES GIMMICK

- **Any mechanic that gatekeeps content a recruiter or researcher needs on the first pass.** Notpron's difficulty is the point of notpron; it is the opposite of the point here. Earned revelation must layer *on top of* immediately legible content, never replace it.
- **XP bars, badges, streak counters, "achievement unlocked" — literal game furniture.** This is the fastest route to AI-tell kitsch; the client's taste profile explicitly forbids decoration-as-art.
- **Forced sequencing or timers** (must-click-in-order, fake loading screens, animated reveals that can't be skipped). A time-loop structure works in Outer Wilds because failure is free and instant; on a CV site it just reads as friction and vanity.
- **JS-dependent puzzle mechanics that break curl-clean HTML.** Every move above must degrade gracefully to plain markup — the moment a mechanism requires a script to mean anything, it has traded machine legibility for a magic trick, which this client considers not a trade-off but a failure.
- **Genre-betrayal used more than once**, or used on a page a stranger arrives at cold (a shared /cv link). The Frog Fractions twist is precious because it is singular; repeated, it becomes a tic, and a tic is kitsch.
- **Wordless restraint applied to the actual prose.** Journey's silence works because Journey has no byte-sacred client-approved text to preserve. Here, restraint belongs to *presentation* — layout, pacing, whitespace — never to cutting or softening the prose itself.
