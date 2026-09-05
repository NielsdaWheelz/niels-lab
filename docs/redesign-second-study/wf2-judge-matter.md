# Judge 3 — The Lens of Matter

## SCORES

1. **tell** 7/10 — Real numbering/cross-ref apparatus is buildable and degrades to plain HTML, but the "accretion" signature moment demands an actual reclassification pipeline that must run forever without ever visibly breaking.
2. **mass** 9/10 — Every mechanism is a build-time function over real numbers (bytes, depth, age) expressed purely in CSS variable-font weight; nothing depends on JS, nothing can visually "break," and reader-mode just strips the one non-essential layer, leaving perfect prose.
3. **ice** 4/10 — Angled wedges meeting at a seam plus a live client-side embedding-similarity call is a demo-reel waiting to happen; no stated fallback for curl/reader-mode, and "live inference" is a runtime dependency this site doesn't need.
4. **source** 8/10 — Technically the cheapest concept in the dossier (HTML comments, zero render cost, zero JS), but its authoring burden — ~480 *genuine* discarded drafts/reverted SHAs/missed lifts — is a research project, not a design system, and will visibly thin out under time pressure.
5. **ephemeris** 7/10 — Solar-time math is well-trodden and the "nothing is locked" clause protects accessibility, but the live 20-second cross-fade on an hour boundary is real client-side engineering that must be gotten exactly right or it reads as broken, not as a sundial.
6. **decay** 6/10 — Buildable if the findings column stays real text and the "film" stays decorative, but the loupe-drop magnifier tied to click position is bespoke interaction work with real accessibility risk if not backed by a text alternative.
7. **excavation** 4/10 — The developed pick (Long Section) demands bespoke hand-inked illustration for hundreds of finds across one giant horizontal scroll — a production and accessibility liability; the discarded **Site Report** candidate (deadpan monograph + Harris Matrix) would have scored 8+ here for being pure structured HTML.
8. **loom** 7/10 — The CSS-custom-property "turn the cloth" flip is genuinely elegant and cheap; the real cost is tagging every entry with which of six facets it "crosses," which is a metadata burden but a sane, finite one.
9. **letter** 7/10 — Zero exotic engineering, degrades to prose and lists perfectly; its only real cost is authorial (writing a slightly-unflattering field note per entry), which is smaller and more natural than source's forensic-provenance demand.
10. **emaki** 3/10 — Replacing scroll with 2D panning across an "infinite" ice plane has no obvious linear fallback for curl, reader-mode, or a screen reader; this is the concept most likely to fail the "text-sovereign" test outright.
11. **homunculus** 8/10 — Opacity/blur driven by real last-edit timestamps and a core-sample slider driven by real git blame are both genuinely cheap, genuinely true data made visible; the versioned-content pipeline for the slider is real work but bounded and valuable on its own.
12. **silence** 9/10 — The entire mechanism is: render everything in markup, hide most of it with CSS/JS, reveal two random lines. It is the single easiest concept to build, the hardest to break, and the one whose curl-vs-browser asymmetry is the argument rather than a risk to it.
13. **prosody** 6/10 — Banding generated from real ledger data is sound, but the glaciological "thin section," calibration sparkline, and per-fleck rendering stack up into a genuinely large custom-visualization build for one person to keep true as data grows.
14. **theater** 8/10 — Almost defiantly unambitious in the right way: a radial-gradient ghost light, restyled lists as callboard notices, a plain sign-in table. Nothing here can fail; it's an engineering safe harbor.
15. **radical** 5/10 — The "no pixel may depict stone" rule is good discipline, but two of its own facets (scroll wheel gaining physical resistance, verse legible only by tilting your head) are accessibility anti-patterns the concept itself proposes without noticing.

## TOP 5

**mass (BEARING).** The only concept where the entire artistic mechanism collapses into a single, inspectable, build-time pure function over real quantities, expressed in CSS. It cannot degrade badly because there's nothing to degrade — strip the styling and you have exactly the austere semantic lists the brief already wants.

**silence (THE THIRD LINE).** Trivial to build, nearly impossible to break, and uniquely honest about the medium: the full corpus sits in the DOM for any crawler or `curl`, while the human gets almost nothing. Under this lens that's not a compromise, it's the safest possible architecture wearing a concept.

**tell.** The register/cross-reference apparatus is just IDs and links — the native grain of HTML — and the "mound" metaphor survives a 60%-shipped version because the registers alone (without the accretion animation) already deliver the whole idea.

**theater (THE GHOST LIGHT).** Deliberately low-tech: a dark field, one lit circle, restyled lists, a plain table. Every single facet is buildable in an afternoon and none of them can regress, which matters more than the metaphor's ceiling.

**homunculus (UNDERCUT).** Two real, cheap, honest mechanisms — CSS derived from real edit timestamps, a slider driven by real git blame — that would be worth building even without the ice conceit. The core-sample versioning pipeline is genuine, bounded engineering work, not decoration.

*Honorable mention:* **source (Foul Papers)** is technically as safe as anything above it — its only real risk is authorial stamina, not engineering.

## GRAFTS

- **mass**: `weight(element) = f(depth, bytes, evidence, age)`, computed once at build and rendered as a variable-font axis — steal this as the site's one universal typographic mechanism regardless of which metaphor wins.
- **homunculus**: list "freshness" (opacity/blur/whatever) derived directly from `git log` last-edit timestamp — a real signal, free to compute, impossible to fake.
- **source**: apparatus that lives entirely in HTML comments — zero render cost, zero JS, rewards only the reader who looks at source, which is exactly the AI-agent/curl audience the brief cares about.
- **silence**: publish the complete structured corpus in the markup regardless of what the visual layer chooses to show — decouples "what a machine can read" from "what a human sees" as a standing architectural rule, not a one-off trick.
- **ephemeris**: the explicit clause that nothing is ever *locked*, only *surfaced first* — whatever selective-disclosure device the final design uses, this is the accessibility escape hatch every one of them needs and only Horarium stated outright.

## THE MISS

None of the sixteen proposed a mechanism for **verifying the metaphor stays honest over time.** Every concept ends with a hand-written "governing rule" — no pixel without a real number, no fracture without a data point, no crack that doesn't trace to a commit — but every one of these rules is *prose intention*, not enforcement. A site that will be maintained by one person for years, under real time pressure, needs its own honesty rules turned into a build-time lint: a CI check that fails the deploy if a disclosure has no attached evidence, if a cross-reference points at nothing, if a computed weight/opacity/depth value has no traceable input, if a claimed git-derived property doesn't actually match `git log`. That's the missing idea: not a metaphor at all, but the thing that keeps every one of these metaphors from quietly rotting into decoration the first time a deadline hits — the mound's own trowel, sharpened into a test suite.
