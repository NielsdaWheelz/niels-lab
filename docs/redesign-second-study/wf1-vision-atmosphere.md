# THE BOOK OF HOURS

*In spring, the dawn. The sky does the moving so the words can keep still.*

## PHILOSOPHY

The site is no longer a book lying on a table; it is a book lying open **near a window**, and the subject of the design is the light that falls on it. Sei Shōnagon's first entry is not a thing but an hour — "In spring, the dawn" — and a Book of Hours is the one manuscript tradition organized by time itself. So the governing metaphor, committed totally: **every page is the same page at a different hour.** Beneath the austere typography lives one painted ground — a Turner sky built from layered translucent washes — whose light is computed, deterministically, from three true facts: the reader's own hour, the season of the calendar, and the recency of the work recorded in the log. This is art and not decoration because the light is never arbitrary: it is the site's honest weather report on a life. A site that trains, ships, and writes has fair days; a site gone quiet clouds over. Turner's move — atmosphere as subject, form dissolving toward light — is here wired to evidence, which is the only way it survives Niels's own moral rule: claim wired to proof, glow wired to fact.

## THE EXPERIENCE

**Arrival, 9:40 pm.** No hero, no card. A deep ink field — but not flat ink: leaning in, the visitor sees it is *weathered*, thirty near-transparent washes deep, murasaki bruising the lower dark like the last light after sundown, a faint diagonal hatch giving the whole ground the tooth of a prepared canvas. Set into this night is a **clearing**: a windless pocket of stabilized dark where the title sits in Newsreader italic at true display scale — the first time this site has ever raised its voice — *The Pillow Book of Niels-Erik Nandal*. The feeling is stepping into a study where one lamp is lit. Nothing moves. Nothing will visibly move all visit.

**The index.** The eight lists stand as the familiar table of hanging em-dash rows — but tonight, because it is evening, *Things that are distant though near* stands open, Shōnagon's own logic: the site keeps her time-index, and the hour chooses which door is ajar. Come back at dawn and it is *Things that quicken the heart*. The reader feels, before understanding it, that the page knows what time it is.

**Opening a list.** The 250ms ink-reveal — the one fast motion the site owns — and the entries pour out. Each list page opens with its first line set large in the clearing, finally allowed the scale the corpus deserves: "Squat bar path: one vertical line, drawn twice." — a sentence given a sky. Below, entries run at reading size, paragraph-indented, book-set.

**Evidence.** The `›` yields, and here light does its most important work: behind the opened proof, a **pool of celadon glaze** gathers — the visual equivalent of bringing the lamp closer to check a citation. On this site the accent color is not brand; it is *the light you read proof by*. The reader feels the difference between assertion and receipt.

**The log.** The ledger's weather is the site's conscience. The most recent row sits in the warmest lamplight on the page; older rows recede up-scroll into cooler, dimmer washes — chiaroscuro as chronology. Struck-through failures sit in an *unlit* band, their attached lessons the only lit text within it. And if the log has gone two weeks silent, the whole page grays toward overcast: deload weather, visible from the doorway. The feeling is standing before something that cannot lie about whether the work is being done.

**The CV.** One step through and the sky evaporates completely — ink on white, dense, one printed page, no atmosphere at all. Walking from a nocturne into a records office. The contrast *is* the point: the reader now knows the weather elsewhere was chosen, not defaulted.

## ART DIRECTION

**The signature move — the Clearing in the Weather.** One living painted ground per view; all text lives in a stabilized pocket whose contrast is clamped to AA no matter what the sky is doing. Atmosphere everywhere, wind nowhere near the words. Every other decision hangs off this.

**Light.** Six canonical hour-states, each a composed painting, not a hue-rotate: **deep night** (near-black ink, murasaki bruise — the canonical state, "night writing" made literal); **cold dawn** (gray-blue, the light of a rink at 6 am, celadon's home); **white noon** (palest state, washes almost dry); **amber afternoon**; **dusk** (violet-gray, the no-JS fallback); **lamplit evening** (warm dark, the reading state). Season tilts the whole continuum — winter cools and darkens, summer warms and lengthens the pale hours. Murasaki and celadon stop being 13px trim and become what they were always meant to be: *the colors the sky passes through.*

**Paint.** Hobbs' watercolor discipline: each state is 25–35 stacked near-transparent SVG washes with jittered edges, seeded deterministically by (date, hour-band, season) — same inputs, same sky, an engineer's weather. One consistent-angle `feTurbulence` hatch (Cézanne's constructive stroke — marks sharing a direction read as *built*) at ≤4% opacity, and a single 2–3% grain varnish unifying text, code, and ground alike. No ruled hairlines anywhere: section boundaries are **plane-shifts** — one wash ending against another, patch against patch, Mont Sainte-Victoire as layout engine.

**Type.** Newsreader and Quattro stay — the faces were never the failure; the timidity was. Newsreader italic finally gets display scale: list titles, the site title, one opening line per list page. Scale logic: exactly three sizes (display in the clearing, body at 68ch, Quattro chrome small), all on one baseline unit. Oldstyle figures in prose, tabular in the log.

**Motion.** The sky's transit between hour-states takes the real hour — Eno's tempo, weather not screensaver; within any sitting, the ground is still. The ink-reveal keeps its 250ms monopoly on fast motion. **Sound: none.** The silence is part of the nocturne.

## THE FACETS

- **Hockey** — cold dawn *is* the rink light: that gray-blue 6 am state every player knows, kept honest and unnamed; and the 404 stands under sodium lamps, the coldest composed light in the system.
- **Strongman** — chiaroscuro as load: in the log, completed work carries the heaviest, most solid ink on the site; the page's darkest darks are *earned tonnage*, and PRs sit fractionally more lit — weight rendered as weight.
- **Novelist** — the emaki scroll: long pages are paced like handscrolls, generous authored gaps (ma as a sized element) between sections, the scroll edited like prose, never dashboard-dense.
- **Poet** — the hour-index itself is the poem: which list opens, which sky attends, follows Shōnagon's clock; structure changes with time while every approved byte stays sacred and still.
- **AI researcher** — the sky ships with a **model card**: an HTML comment atop every page and a colophon table publishing seed, parameters, and state ("sky: 2026-09-04 · evening · autumn · seed 8121") — a generative system that shows its weights.
- **Engineer** — total determinism: the atmosphere is a pure function of clock, calendar, and log timestamp; no randomness at runtime, reproducible to the byte, curated Molnar-style from generated batches before shipping.

## SIGNATURE MOMENTS

1. At 4:47 am the site is at its deepest ink and *Things that quicken the heart* stands open — the design's whole thesis in one dawn visit.
2. Opening an evidence mark pools celadon lamplight behind the proof: on this site, light is what you check citations by.
3. The hidden draft reading list appears non-finito — its first entries set clean, the rest dissolving mid-list into raw hatched ground, Rodin's figure still in the stone, honest about its unfinishedness.
4. Two silent weeks in the log and the ledger clouds over; the reader who returns after a real deload sees real weather.
5. View-source rewards the curious with the sky's model card and a signed colophon comment — the page is *more* composed underneath, never less.
6. Print the CV and every wash evaporates: pure ink, one page — proof the atmosphere was a choice with an off-switch, not a coat of paint.

## MACHINE SURFACE

The semantic HTML is untouched and identical to today's: curl, reader mode, and crawlers receive the complete, correct document — clearings, skies, and glazes live entirely in a CSS/SVG layer painted over it. One small inline script (confessed in the colophon, joining the two permitted client components) reads the clock and sets custom properties; with JS absent the site renders its canonical **dusk** — a complete, composed painting, not a degraded one. The current hour-state is exposed as a `data-sky` attribute and the model-card comment, so agents can *read the weather* — machine legibility gains a fact rather than losing one. llms.txt, RSS, sitemap, JSON-LD, canonical bio: byte-identical pipeline. The OG card is repainted in the current season's sky, so even the link preview keeps the hour. Evidence validation, zero-JS parity, print CV: unbroken.

## SACRIFICES

- **The whole palette, never at once.** A single visit shows one hour; the design's richness exists only across returns. Right price: this is a site for the returning reader, and a painting you can exhaust in one look is a poster.
- **The manual light/dark toggle as a binary.** It becomes an hour dial — the reader may set the site to any hour, which subsumes light/dark and keeps accessibility control while deepening the metaphor.
- **The six-token austerity.** Tokens become a computed continuum with clamped contrast invariants. More machinery, but the old austerity was the diagnosed failure.
- **Screenshot stability and some page weight** (wash layers budgeted under 60KB, inlined). The site will never look the same twice; that is the feature.

## GRAVITY

**Hard to build:** six skies that look *authored* — the difference between painting and preset is thirty tuned layers per state, curated from batches, weeks of taste-work; AA across a continuum solved only by the Clearing's clamped-contrast invariant, which must be enforced in the token math and tested per state; SVG paint cost on low-end devices (mitigate: static composited layers, zero runtime filters after first paint). **What could go wrong:** the sky winning the first half-second from the words — contrast miscalibration is fatal and must be tested at every hour-state; the log-weather reading as gimmick if its thresholds feel arbitrary (they must be published in the colophon, like everything else here). **Where the kitsch lives and the police that hold the line:** the lava lamp. Standing orders — one atmospheric idea per surface, ever; nothing but the ink-reveal moves faster than minutes; no cursor-glow, no parallax, no pulse; and the golden rule with no exceptions: **every unit of light traces to a real datum — the clock, the calendar, the log — or it is cut.** A glow that means nothing is a gradient with a alibi. Here, nothing glows without a reason, which is exactly what makes the whole site, finally, sensuous: it is not decorated. It is *lit*.
