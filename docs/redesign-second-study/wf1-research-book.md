# Book Design Traditions → Web Mechanisms

## TRADITIONS

**Bringhurst / webtypography.net** — *Elements of Typographic Style*, ported to CSS by Richard Rutter's [webtypography.net](https://webtypography.net/). Mechanism: measure (line length) as the load-bearing variable — 45–90 characters, ideally ~66; baseline rhythm (vertical grid all elements snap to); type as a "musical" system where size, leading, and measure are tuned together, not set independently.

**Van de Graaf canon / Tschichold** — the "secret canon" construction: page divided so margins run 2:3:4:6 (inner:top:outer:bottom), text block always echoes the page's own proportion (2:3, or golden section). Mechanism: margins are *derived*, not padding — the outer margin is deliberately the largest, reserved space, not leftover space. [Canons of page construction](https://en.wikipedia.org/wiki/Canons_of_page_construction).

**Fine press — Kelmscott, Doves, Officina Bodoni** — two opposite poles from the same movement. Kelmscott Chaucer (Morris, 1896): maximalist unity — type, border, and woodcut illustration designed as one inseparable page-object. Doves Press (Cobden-Sanderson/Walker, 1900): the opposite discipline — "austere... eschewing all decoration," the page speaking through type, spacing, and presswork alone. Officina Bodoni (Mardersteig): revival-grade craft precision without ornament. Mechanism: beauty from *unity of system*, not from added decoration — either richly unified ornament or nothing, never decoration bolted onto a generic grid.

**Rubrication** — red ink as a second, structural channel in medieval manuscripts: marks section boundaries, headings, liturgical instruction, distinguishing what's *procedural* from what's *read*. Not decorative — a semantic color with one job. [Rubrication](https://en.wikipedia.org/wiki/Rubrication).

**Talmud page / commonplace books** — the Talmud's page architecture puts core text centered, Rashi's commentary on the inner margin, Tosafot on the outer, cross-references above — "the original hypertext," annotation literally surrounding evidence rather than being hidden below it ([Tikvah Ideas](https://ideas.tikvah.org/mosaic/picks/how-and-why-the-talmud-got-its-distinctive-look)). Commonplace books are the private-reader mirror: a compiled apparatus of quotation plus the reader's own gloss, built to be re-entered and cross-referenced, not read once linearly.

**Heian kana / chirashigaki / ma / emaki** — chirashigaki ("scattered writing"): waka poems set with column lengths deliberately varied to create rhythm and an "attractive composition," sometimes breaking mid-word — asymmetry as *controlled* compositional device, still perfectly legible in sequence ([Wikipedia: Chirashi-gaki](https://en.wikipedia.org/wiki/Chirashi-gaki)). Ma: negative space as an active, weighted element — "an emptiness full of possibilities" — not a gap between content but content in its own right. Emaki handscrolls: narrative unrolled slowly, revealed in time rather than surveyed all at once.

**Keats manuscripts** — the draft as visual object: crossed-out lines, cramped margins, the "living hand" (Ode to a Nightingale survives only in his own scrawled hand). Mechanism: revision *visible* is itself expressive — process as content, not something cleaned up before publication.

**Modern web practitioners** — Butterick: measure discipline (45–90 characters, scale font-size and column together on resize). Tufte-CSS / Gwern: sidenotes replacing footnotes (marginal, not bottom-of-document; collapse to inline on narrow viewports), margin notes as skim-aids, epigraphs, sparing dropcaps, link-type icons as a second referential layer.

## THE TRANSLATION — 12 moves

1. **Canon-derived margins, not symmetric padding.** Compute the type column with a Van de Graaf/Tschichold-style ratio (e.g., asymmetric 2:3:4:6) instead of equal padding on all sides. The oversized outer margin isn't wasted space — it becomes the reserved zone for #3.

2. **Measure as a hard constraint.** Cap prose at 60–72ch using the `ch` unit; never let body text run full-bleed on wide screens. This is the single highest-leverage move from Bringhurst/Butterick.

3. **Sidenotes, not footnotes**, for CV citations and evidence links (Tufte/Gwern mechanism). On wide viewports, notes sit in the reserved outer margin at the height of their reference; on narrow viewports they collapse to inline or a tap-to-reveal — never a bottom-of-page dump. This is close to load-bearing for a CV that is "wired to primary evidence."

4. **One rubrication thread, not a palette.** Reserve a single red (or whatever accent) for exactly the jobs rubrication did: section markers, the drop cap, hover state on primary links. If the accent starts appearing on buttons, backgrounds, or icons too, it has stopped being structural and become decoration — the exact failure mode of AI-slop gradients.

5. **Real drop caps, sparingly.** CSS `initial-letter` (with a `::first-letter` float fallback for unsupported browsers) at major section opens only — never per-paragraph. Set it in the body typeface, not a separate "decorative" font; a drop cap in a different face reads as costume.

6. **Hanging punctuation as progressive enhancement.** The `hanging-punctuation` CSS property (Safari only today) plus a light polyfill for quote marks and list bullets — this is exactly where Shonagon-style bulleted lists get an optically straight left edge for free.

7. **Chirashigaki-style asymmetry for verse/pull-quotes.** Where a poem excerpt or epigraph appears, let line indentation and vertical placement vary deliberately rather than force it into the body grid — but keep it strictly legible in reading order. Controlled irregularity, never randomness.

8. **Ma as a sized element, not filler.** Give section-break whitespace an explicit rule (e.g., a chapter divider is N lines of pure space plus one small glyph) so negative space is authored, not just "margin: auto" leftover.

9. **Scroll-as-emaki for long-form pages.** Treat the CV/essay page as a handscroll unrolling in time: generous, paced vertical gaps between sections, content revealed progressively — the opposite of a dense dashboard grid. This also directly supports the "poetry, not utility" mandate.

10. **A Talmud-style commentary lattice for evidence-wired entries.** Primary claim/entry in the main column; citation, date, or link apparatus in a fixed side column that runs parallel to it (not a hover tooltip, not a footnote) — core text and commentary visibly co-present, the way Rashi and Tosafot flank the Talmud text.

11. **A single baseline unit governing everything.** Pick one line-height-derived grid unit; force headings, rules, images, and section spacing to land on multiples of it. This is the quiet coherence Bringhurst calls "harmony" — invisible when done, jarring when skipped.

12. **Manuscript-honesty for process/draft material.** If the site ever shows a writeup's evolution, show real diffs/marginal revision notes (Keats mechanism) rather than a fake-parchment "old paper" skin. Process shown honestly reads as craft; process costumed as artifact reads as kitsch.

## TRAPS

- **Faux parchment / paper-grain textures, ink-bleed filters, fake foxing.** This is the exact skeuomorphic kitsch this client already named as failure mode #1 — the print equivalent of a purple gradient. The manuscript traditions above are sources for *structure and rhythm*, never for surface texture.
- **Decorative drop caps as clip-art.** A drop cap rendered in a separate illuminated-manuscript font or as an illustrated image reads as costume, not lineage. Keep it in the body typeface; let scale alone do the ceremony.
- **`hanging-punctuation` has near-zero browser support** (Safari only). Build it as pure enhancement; never let layout depend on it rendering.
- **Rubrication creep.** The instant a "structural" red starts decorating things that aren't structural markers, it has become a brand-color accent — indistinguishable from the AI-tell palette this whole exercise exists to avoid.
- **Van de Graaf literalism on a responsive viewport.** The canon was built for a fixed sheet size; port the *logic* (derived, asymmetric margins; a reserved outer zone) not the exact geometric construction, which doesn't survive reflow.
- **Sidenotes with no real mobile behavior.** A sidenote that's simply `display:none` below some breakpoint deletes the evidence apparatus that makes the CV credible. It must degrade to inline or disclosed, never vanish.
- **Ma/chirashigaki asymmetry mistaken for sloppiness.** Historical scattered writing is controlled and stays legible in sequence; uneven-for-its-own-sake indentation reads as broken CSS, not as hand. Every asymmetric placement needs a rule behind it, even an invisible one.
