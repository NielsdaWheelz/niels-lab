# The register

How prose is written on nielseriknandal.com. This governs every word on the site —
page copy, project writeups, essays, list entries, meta descriptions, CV bullets —
except the four frozen posts (zero-to-hero-one, zero-to-hero-two, my-first-project,
fractal-go), which are historical record. The approved corpus (`src/content/lists.ts`
and the colophon) is the voice anchor: where this document and the corpus disagree,
the corpus wins and this document gets amended.

Lineage: Sei Shōnagon supplies the form. Montaigne supplies the stance — his candor,
not his syntax; a Montaigne-length sentence is a bug. Orwell supplies the sentence
discipline. Dan Luu, Julia Evans, Simon Willison, Patrick McKenzie, Brandur Leach,
Fabien Sanglard, and the Jane Street blog supply the evidence mechanics.

## The voice

Six sentence shapes carry the site. Use them; do not invent new ornaments.

1. **Image, then verdict.** A concrete thing, then flat judgment. No connective tissue.
   "A mock of the unit under test. The suite is green because it agrees with itself."
2. **Colon definition.** Subject, colon, compressed predicate that does the arguing.
   "Squat bar path: one vertical line, drawn twice."
3. **Mirrored clause.** The judgment lands as symmetry or reversal.
   "Weight tying: the matrix that reads is the matrix that speaks."
4. **Rule, then cost.** Mechanism stated as consequence, never as advice.
   "One concern per revision, so the rollback is a sentence instead of a meeting."
5. **Freestanding fragment.** A noun phrase with one qualifying clause is a complete entry.
   "Sleep, with the diff still open."
6. **Plain SVO** for explanatory prose: premise before consequence, parallel items on
   semicolons, terms defined before reuse.

Rhythm: long unit, then short. The shortest sentence ends the paragraph and carries
the verdict. Entries run 5–25 words; the second sentence is shorter than the first.

Punctuation: the em-dash is spaced ( — ), at most one per sentence, and does two jobs
only — appending an inventory, or a late turn. The colon defines. The semicolon joins
parallel clauses. The period is the default joint; prefer a new sentence to a
conjunction. Zero exclamation marks. Questions only as quoted speech or section
headers that data answers.

Capitalization: page titles and post headings lowercase; list titles sentence case;
proper nouns exact (RMSNorm, Pydantic, Sei Shōnagon). No bold or italic as emphasis;
italics for work titles only.

Vocabulary: concrete nouns from the home domains — barbell, rink, long books, model
internals — unglossed, beside plain verbs. Numbers exact and unrounded: "182
hand-written Alembic migrations", "p = 0.051". Adjectives scarce, physical or moral,
never promotional. Intensifiers do not exist here.

Judgment: asserted as fact with the mechanism attached, owned by "I", never softened.
Self-assessment is deflationary and literal: "It is not built, and this paragraph is
currently its entire implementation."

## Hard rules

1. **No metaphors, no similes.** Stock or fresh. A figure asks the reader to decode
   an image instead of receiving a fact. Say the thing itself.
   Before: "PDFs are hostile." After: "The two extractors disagree on 4% of PDFs;
   the fallback picks the longer text."
2. **One idea per sentence, stated once.** If a sentence restates a prior sentence,
   delete the second telling. If it needs a semicolon and two commas, make it three
   sentences.
3. **One idea in one place, site-wide.** The writeup is the authoritative long form;
   list entries, CV bullets, and meta descriptions compress and link. A post never
   competes with an approved list line — the list keeps the phrasing, the post loses it.
4. **Active voice; the actor is load-bearing.** Who failed, decided, fixed. Passive
   only when the actor is genuinely unknown or irrelevant.
5. **Short words.** use, not utilize; help, not facilitate; method, not methodology;
   fast, not performant; lessons, not learnings. Cut: basically, actually, very,
   really, "it is worth noting", "in order to", "going forward".
6. **No marketing.** No promotional adjectives (comprehensive, full, clean, modern,
   robust, seamless, real as intensifier), no audience flattery, no deck-speak, no
   direct pitch to a hiring reader. The enumeration proves the scope; the adjective
   only weakens it.
7. **No hedging as armor.** No "I'm no expert, but", no pre-emptive self-defense, no
   disclaimers against accusations nobody made, no defensive negation before a claim
   ("this is not a toy"). Limits are stated as fact (see Stance).
8. **Open on the thing.** No throat-clearing, no context-setting preamble, no
   announcing intent, no meta-commentary headings ("the idea in one paragraph").
9. **Stop when done.** No summary morals, no "in conclusion", no calls to action.
   End on the last concrete item or a one-line verdict.
10. **Clarity outranks compliance.** If the terse version is ambiguous, add the words
    that remove the ambiguity. Terse means no waste, not cryptic.

## Stance

- First person. "I" chose, built, broke, and misjudged; no corporate "we", no
  "one might consider".
- Self-scrutiny is the method: the writer's own judgment is part of the material.
  "I distrust this design partly because I designed the last one, and I remember it."
- Candor about failure is required, not optional. Every writeup says what is wrong
  with the thing. Postmortems where the author is never the cause are fiction.
- Limits stated as fact, not disclaimer: "I have never operated Kafka at scale.
  Within that limit: the tutorial defaults are wrong."
- Precision-hedging is licensed; vague weaseling is not. Distrust your own setup,
  name the suspected cause, mark the guess: "I suspect the tokenizer cache, but I
  have not isolated it." Never "arguably", "sort of", "I could be wrong but".

## Evidence

- Links sit on nouns, mid-sentence — the commit, the benchmark, the demo. Never
  "see here", never "[link]", never a bare URL dump in prose.
- One number per claim; ranges over adjectives. "2–20% overhead, under 10% observed"
  beats "low overhead".
- State the evidence's provenance and size before interpreting it: n, method, date.
- A claim without public evidence goes as plain observation, unmarked. No placeholder
  links, ever; an evidence link pointing at a draft or a dead page fails the build.
- Credit prior art by name, inline, the moment it is relevant.

## Forms

**List entry.** 5–25 words, one to two sentences, one of the six shapes. A titled
list is a complete piece: no intro, no conclusion. Ends when the observation ends.

**Project writeup.** 300–600 words. Must answer, in order or woven: what it is (one
line); why it existed (the itch, one sentence, no throat-clearing); the one to three
hard decisions, each with the alternative named and evidence on the nouns; the
numbers, with provenance; what is wrong with it, as fact; a one-line close — verdict
on the opening itch, the principle served, or a standing offer of counter-evidence.
Scope cut is named as a decision, not an apology. Headers optional; if used, they are
questions the content answers.

**Essay / journal.** Shōnagon's rules: open mid-world on the object, concrete
particulars, categorical judgment, stop when done. Length is whatever the material
fills, in the six shapes.

**Connective copy** (meta descriptions, page intros, /log furniture, footer). One
sentence where one sentence does; meta descriptions describe the page for an index
and never duplicate the body intro verbatim; furniture states what the surface shows
and what its absence means, without inventing causes.

**CV bullets.** Verb first, then the object, then the consequence — one idea per
bullet. "Built", not "architected". No intensifiers; the stack list is the evidence.

## Not banned

- **Terms of art.** pipeline, branch, thread, cache, race, deadlock, hot path, cold
  start — the standard name for a concept is vocabulary, not metaphor. The rule bans
  invoking the image: "we deleted forty stale branches" is fine; "our branches grew
  into a thicket" is not. Coining a figurative name is metaphor; using the industry's
  fixed name is not.
- **Calibrated technical qualifiers** tied to real variance: "with very small batches,
  the statistics can be noisy enough to make training unstable." That is precision,
  not hedging.
- **The corpus's own device** — compressed personification of the author's artifacts
  ("The colophon confesses."; "Byte fallback, keeping its promise.") — is licensed in
  the list register and the colophon only: one figure maximum per entry, never
  sustained across sentences, never elsewhere. In writeups, essays, and connective
  copy it is banned with the rest.

## Pre-publish check

1. What am I trying to say — can I point at the one idea?
2. Does it open on the thing itself?
3. Is every noun a particular; is every number exact?
4. Have I said anything twice — on this page, or anywhere on the site?
5. Have I written an image instead of a fact?
6. Are judgments flat and owned by "I"; is at least one failure or limit stated as fact?
7. Could each sentence be read aloud in one breath?
8. Does every checkable claim link to primary evidence on its noun — and every
   uncheckable one read as plain observation?
