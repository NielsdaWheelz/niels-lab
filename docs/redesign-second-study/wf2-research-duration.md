# Durational, Ambient & Sounded Media — Research Brief

## SOURCES

**Brian Eno — 77 Million Paintings** — Software recombines 296 hand-painted slides four at a time, glacial crossfades, paired with generative sound; the "77 million" is just the count of combinations, so nothing repeats in any single visitor's lifetime. Mechanism: durational meaning from *recombination of a fixed, finite set*, not from continuously authored new content. [Wikipedia](https://en.wikipedia.org/wiki/77_Million_Paintings) · [Long Now](https://longnow.org/ideas/10-years-ago-brian-enos-77-million-paintings-in-san-francisco-02007/)

**Brian Eno — *Music for Airports* liner notes** — Coined "ambient": music that "must be able to accommodate many levels of listening attention... as ignorable as it is interesting." This is the load-bearing design test for everything below. [uDiscoverMusic](https://www.udiscovermusic.com/stories/brian-eno-music-for-airports-feature/)

**Longplayer (Jem Finer)** — Six looping source recordings, restarted from calculated offsets every two minutes; simple modular arithmetic guarantees no combination repeats for exactly 1,000 years. No new material is ever generated — duration comes entirely from combinatorics applied to a small fixed corpus. [longplayer.org](https://longplayer.org/about/how-does-longplayer-work/) · [Artangel](https://www.artangel.org.uk/longplayer/how-longplayer-works/)

**John Cage — *Organ²/ASLSP* at Halberstadt** — A 639-year performance; the organ is physically built out pipe by pipe as the piece proceeds, and chord changes land years apart (next: October 2027). The instrument's incompleteness *is* the artwork's visible clock. [Classic FM](https://www.classicfm.com/composers/cage/as-slow-as-possible-organ-chord-change/) · [Halberstadt.de](https://www.halberstadt.de/en/john-cage-organ-project.html)

**John Cage — *4′33″*** — Three timed movements of no intentional sound, forcing attention onto ambient/incidental sound already present. Duration + a ritual frame turns background into foreground without adding anything. [Britannica](https://www.britannica.com/topic/433-by-Cage)

**La Monte Young & Marian Zazeela — Dream House** — Permanent sine-wave drone installation (since 1993) where the sound changes with the listener's *position*, not with time: move your head six inches, a different harmonic dominates. Duration here is spatial, not sequential. [In Sheep's Clothing](https://insheepsclothinghifi.com/dream-house-installation/)

**Mark Weiser & Amber Case — Calm Technology** — Attention lives mainly in the periphery; the technology should be able to move between center and periphery at will and "communicate but not demand." A vocabulary for how much a durational/ambient element is allowed to ask of a reader. [Wikipedia](https://en.wikipedia.org/wiki/Calm_technology) · [Case Organic](https://www.caseorganic.com/post/principles-of-calm-technology)

**Clock of the Long Now (Danny Hillis)** — Ticks once per minute; chimes a unique bell sequence every day for 10,000 years without repeating; design principles are longevity, maintainability, transparency, evolvability. "Pace layers" — fashion fastest, nature slowest — is a reusable frame for deciding what on a site should move fast vs. never. [Long Now](https://longnow.org/clock/faq/)

**Maarten Baas — *Real Time*** — A person, on video, hand-paints and erases clock hands once a minute, filmed in true real time. Duration is made *costly and visible* — the opposite of a CSS animation — which is the honesty move worth stealing conceptually even without literally filming a human. [Wikipedia](https://en.wikipedia.org/wiki/Real_Time_(art_series))

**Patatap (Jono Brandel + Lullatone)** — Keystroke-triggered sound+shape, built on Web Audio; every event is instant, discrete, decaying, and entirely user-initiated — nothing plays unless a key is pressed. [CreativeApplications](https://www.creativeapplications.net/project/patatap-portable-animation-and-sound-kit-by-jonobr1-and-lullatone/)

**ambient.garden** — Generative audio "landscape," shipped in two parallel modes: a pre-rendered "frozen" version (full interactivity, static loops) and a live code-driven version. The static mode is a first-class, indistinguishable fallback, not a degraded one. [GitHub](https://github.com/pac-dev/AmbientGarden)

**Japanese incense clocks (kōdokei)** — Time told by scent changing as different aromatic segments burn — a second sense layered onto a single slow physical process, more accurate than water clocks because combustion is steady. [Urchin's Home](https://urchinshome.com/blogs/stories/incense-clocks-to-measure-time-set-the-timepiece-on-fire)

**Slow Web (Jack Cheng et al.)** — Deliberate latency as authored choice: iDoneThis processed email once a day and told users so at the bottom of every message. Batching, not streaming, is the design move. [I Done This](https://blog.idonethis.com/the-slow-web-movement/)

## STEALABLE MOVES

1. **Recombine, don't add.** Longplayer/77M Paintings generate century-scale duration from a *fixed* small corpus. Apply this to the zuihitsu lists and /log: a deterministic, day-seeded reorder or emphasis-rotation of the existing ~60 entries (pure server-side or CSS `nth-child`/`:target`, no client JS) so the same byte-sacred content reads differently over time without a single new sentence being written.

2. **Make the update cadence itself an artifact.** Halberstadt's organ is visibly under construction; each pipe change is a scheduled, published date. Put a genuinely rare-cadence element on the site — one line, one color token, one artifact — with a stated next-change date, so persistence (not churn) reads as the signal of seriousness.

3. **Silence as a frame, not an absence.** 4′33″'s trick is a ritual container that redirects attention to ambient content already there. Give one page (e.g. /now or /colophon) a deliberately near-empty treatment — no ornament, maximal negative space — so the reader's own room and reading pace becomes the "sound."

4. **Position over time.** Dream House changes with where you stand, not when. Tie one subtle visual variance (a hairline color, a type-weight shift) to scroll depth or viewport rather than a clock — durational without a single running timer.

5. **Sound only as instrument, never as ambience.** If any audio ever ships, follow Patatap's contract exactly: user-struck, single, decaying, never looping, never on load. This is the only sound posture compatible with "correct but not beautiful is the failure" and with WCAG 1.4.2.

6. **Ship the static twin first.** ambient.garden's frozen/live pairing means the generative layer is optional garnish over a complete, motionless page. Any experimental durational feature needs a fully legible, zero-JS fallback that a crawler or `curl` sees as the *real* page, not a degraded one.

7. **Batch the ledger.** Apply Slow Web literally to /log: commit to a stated cadence (end-of-day, end-of-week) rather than live-updating, and say so in one sentence near the ledger — latency as an authored, disclosed choice rather than an infrastructure limitation.

8. **Motion measured in minutes, not seconds.** Eno's crossfades and the 10,000-Year Clock's one-tick-per-minute set the tempo ceiling: any ambient CSS animation on the site should cycle on the order of tens of minutes, invisible in a normal visit, so it satisfies "ignorable as interesting" rather than "eye-catching."

9. **Second-sense time-telling.** The incense clock marks time through a channel other than a numeral countdown. A footer note like "last edited: [date]" or "N days since last entry," server-rendered once and static per load, does the same job as a ticking clock without ticking — calm technology's periphery principle made literal.

## THE OBNOXIOUSNESS LINE

Autoplay of anything with sound is not a design risk, it is a violation — WCAG 1.4.2 exists because starting audio without consent is coercive, and Eno's own standard already forbids it: a sound that seizes the center of attention has failed as ambient music. The test that separates the whole list above from slop is Eno's sentence, applied literally: *if a reader would notice the mechanism only by staring, it belongs; if they'd notice it by being interrupted, it doesn't.* Corollaries: motion perceptible within a single scroll (anything faster than a multi-minute cycle) reads as "modern web," not art. Anything that makes the reader wait — a loading ritual, a forced interval, a countdown before content appears — has inverted calm technology's promise and turned duration into a toll. And any durational/sound element that only works once, or that requires the visitor to "get" a joke to tolerate it, has already crossed from Cage into gimmick — the line isn't cleverness, it's whether removing the element changes what the *content* means (fine, essential even) or only what decorates it (the tell).
