# CANDIDATES

1. **Foul Papers** — the rendered page is the fair copy and the HTML source is the foul papers, so "view source" surfaces the actual struck-through drafts beneath the byte-sacred final lines.
2. **The Boustrophedon Ledger** — the page reads like a plowed field, each line reversing direction as the ox turns, so scrolling re-enacts the turn-and-return of training, drafting, and revision.
3. **The Case File** — the whole site is formatted as a forensic dossier on a still-living subject, lists as exhibits, essays as testimony, cross-referenced by evidence numbers instead of nav links.
4. **The Ice Core** — the site is a drilled core sample; scrolling descends through compressed years, and each list is a stratum you must decompress to read what's frozen in it.
5. **The Opening Repertoire** — the record of a life catalogued like a chess opening book, ECO-style codes and "!?"/"?!" glosses annotating his own decisions, because he is always mid-game.

---

# Foul Papers

**Epigraph:** *Every fair copy hides its foul papers.*

**THE IDEA.** The rendered site changes nothing — same austere Newsreader page, same eight lists, same dark quiet. What changes is what the page *is*, structurally: a fair copy, the clean transcript a compositor sets from a messier original. Its HTML source is that original — the foul papers. Comments hold struck lines. IDs are printer's signature marks (`sig-C3v`, not `entry-14` — a real bibliographic convention: gathering, leaf, recto/verso). In the first sixty seconds a visitor reads the lists exactly as before. Then, in a footer line or the colophon, one sentence: *this page has two texts; the second is beneath.* Curiosity does the rest.

**HOW THE LISTS LIVE IN IT.** Each one-line entry keeps its existing zero-JS disclosure to evidence — untouched. But the markup around it now carries a second, source-only apparatus: an HTML comment holding that entry's actual earlier draft, struck through in the comment's own text, ending in the kept line. The disclosure still opens to real evidence in the rendered page; the comment opens, to whoever looks, onto the labor that produced the sentence. Nothing rendered is added or removed — the byte-sacred copy stays sacred because the marginalia lives where no renderer looks.

**SIGNATURE MOMENT.** You view-source on an ordinary list item and find:
`<!-- 1st hand: "I have broken more noses than I've written poems." / struck: "brutal" → kept: "exact" -->`
— above the line that actually renders: *"Things that are exact: a wrist shot, a deadlift, a line of iambic pentameter."* The public sentence is suddenly legible as a choice, not a given.

**WHY ART, NOT GIMMICK.** Every public self here has a private draft: the game and the practice reps, the successful lift and the missed ones, the published book and its rejected pages, the finished poem and the discarded stanza, the paper and the failed run, the shipped commit and the reverted one. The form doesn't illustrate that fact, it *is* that fact — the site's own visible/invisible split is the exact shape of a working life. It survives visit ten because the foul papers are never finished: new struck lines get added as the actual life adds more of them. It's a living manuscript, not a one-time reveal, and machine legibility is served, not spent — signature-mark IDs are stable, semantic, curl-clean; the comments cost nothing to a parser and everything to a reader.

**THE FACETS.**
- *Hockey player* — entry signatures run on period-and-clock notation (`¶2.14`) instead of generic anchors.
- *Strongman* — missed lifts appear as struck weights in comments beside the kept PRs; the record of failure exists only in source.
- *Novelist* — essays carry a variorum apparatus: footnoted alternate endings that never reached the fair copy.
- *Poet* — a poem's rendered line breaks correspond, in the comment above it, to the actual physical line breaks of the handwritten first draft.
- *AI researcher* — marginal "errata" comments hold the real failed prompts and wrong hyperparameters behind a stated result.
- *Engineer* — reverted commit SHAs sit as comments beside the feature that replaced them — a graveyard visible only to those who look underneath.

**THE RISK.** Kitsch arrives the moment "ye olde manuscript" becomes costume — invented archaisms, a puzzle-hunt that rewards cleverness with nothing but cleverness. The rule: every mark in the source must be true. If it isn't an actual discarded line, an actual reverted commit, an actual missed lift, it does not go in the comment.
