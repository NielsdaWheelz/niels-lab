## CANDIDATES

1. **The Third Line** — two unrelated list-fragments per visit, cut by a timed span of dark; a human never sees the whole archive, but any crawler reading the markup gets everything.
2. **The Sitting** — the page fills with text only as long as the visitor stays still, and drains back to black the instant they move, scroll, or click.
3. **The Goban** — a blank 19×19 grid; resting the cursor on one point for several unbroken seconds reveals a single entry forever, so the site fills communally, over years.
4. **The Bell** — one glyph; striking it releases an entry riding a real bell's decay curve, gone exactly when the sound is, and it will not repeat within a sitting.
5. **The Necrology** — the site as a one-page death notice under solid black redaction bars, dragged aside like gravestones to read, then falling shut again.

---

## THE THIRD LINE

*"Most of what I am is the part I didn't say."*

**THE IDEA.** The page is black. One line rises — an entry pulled at random from across the eight lists ("Things That Give No Warning: a torn labrum, a manuscript's last sentence, love"). It holds, then the field goes dark for a fixed, authored span — not a spinner, a rest — and a second, unrelated line rises from a different list. Dark again. One option: stay, or reload for a different pair, on the same clock. No nav, no scroll, no archive view. That is the visit.

**HOW THE LISTS LIVE IN IT.** Nothing is deleted. All eight lists, every dated entry, every zero-JS disclosure, sit whole in the document — the data the draw pulls from, not decoration over it. The only rule: the two lines must come from different lists. So every load forces the site's real subject — a hockey injury beside a line of verse beside a merge conflict — to perform itself once, structurally, whether anyone notices or not.

**ONE SIGNATURE MOMENT.** Someone runs `curl` on it out of curiosity and the whole thing is there — years of lists, fully marked up, plain in the source. Open it in a browser instead and you get two lines and a lot of dark. The machine was told everything. The person was told almost nothing.

**WHY ART, NOT GIMMICK.** The asymmetry is the argument, not a trick with a solution. A life read linearly and completely, the way a machine reads it, isn't the same thing as a life known, which arrives in fragments, slowly, across returns, held together by the visitor's own memory rather than the page's. There is no puzzle and no unlocking; each visit is a different, unrepeatable pair, so nothing about it can be solved and shelved. It survives visit ten the way weather does.

**THE FACETS.**
*Hockey player* — the cut between lines is hard, no fade: the whistle stops play, it doesn't dissolve it.
*Strongman* — the dark spans lengthen in small increments the more a visitor has seen (tracked only locally, never server-side), so return visits carry more, never less.
*Novelist* — entries don't repeat within a rolling window; the same sentence isn't allowed back too soon.
*Poet* — the only punctuation permitted between the two lines is the blank itself; no ellipsis softens the cut.
*AI researcher* — the fullest, most rigorously structured archive he has published exists only for the reader that never needs to see it rendered.
*Engineer* — the withholding uses no obfuscation, visible in any network tab; the restraint is taste and CSS, not a locked door.

**THE RISK.** The failure mode is the pairing engine getting clever — matching lines for false resonance until the site reads like a fortune-cookie oracle. The governing rule: the draw is a coin flip constrained only by category, never by sentiment. Any resonance felt between two lines is work the visitor's own mind did. The moment the system tries to mean something on its own, it has become decoration.
