# Painting and Sculpture on Screen

## SOURCES

**Turner (late oils, sea pictures, "indistinctness is my forte")** — atmosphere *is* the subject, not backdrop to it. Form dissolves toward light; the sublime lives in what won't resolve into a clean edge. [Tate: Sea Pictures](https://www.tate.org.uk/art/research-publications/the-sublime/david-blayney-brown-sea-pictures-turners-marine-sublime-and-a-sketchbook-of-c1803-10-r1141418) · [Tate Etc: Unfinished? Repulsive? Or the work of a prophet?](https://www.tate.org.uk/tate-etc/issue-15-spring-2009/unfinished-repulsive-or-work-prophet)

**Cézanne (constructive stroke, Mont Sainte-Victoire series)** — brushstrokes at a consistent diagonal, laid down *irrespective of underlying form*, building solidity through repetition and color-plane shift rather than outline. Structure as accumulated marks, not a drawn boundary. His "doubt" — the unresolved patches he saw as failure, we can read as honesty about perception mid-process. [Eclectic Light: Cézanne and constructive strokes](https://eclecticlight.co/2015/11/17/trees-in-the-landscape-6-paul-cezanne-and-constructive-strokes/)

**Rodin (non-finito, partial figures, *La Pensée*)** — the figure surges out of unworked stone; incompleteness is philosophical, not a failure to finish (Roman sculptors signed work *faciebat* — "was making," present tense, forever). [Wikipedia: Non finito](https://en.wikipedia.org/wiki/Non_finito)

**Vera Molnar** — randomness as "artificial intuition": generate widely, then converge by hand-picked judgment. The algorithm proposes; the artist still decides. [Sotheby's: Vera Molnár](https://www.sothebys.com/en/articles/vera-molnar-the-grande-dame-of-generative-art)

**Tyler Hobbs (watercolor algorithm, Fidenza)** — painterly texture comes from 30–100 nearly-transparent, edge-jittered layers stacked, not one clever gradient formula. Flow fields give *one* structural idea enough range to surprise without ever looking uncomposed. [A Guide to Simulating Watercolor Paint](https://www.tylerxhobbs.com/words/a-guide-to-simulating-watercolor-paint-with-generative-art) · [Fidenza](https://www.tylerxhobbs.com/words/fidenza)

**Anders Hoff (Inconvergent)** — "enough order to be recognizable, enough chaos to break out of ordinary forms." Simple rules, legible outcome. [inconvergent.net](https://inconvergent.net/)

**Brian Eno, *77 Million Paintings*** and **Jem Finer, *Longplayer*** — slowness as the content: generative change paced so slow (Eno: 400 years to repeat; Longplayer: 1,000 years, no loop) that an attentive viewer never catches it "animating." The couches, not the screen, are the design move. [Long Now: 77 Million Paintings](https://legacy.longnow.org/events/02007/jun/29/77-million-paintings-brian-eno/) · [Longplayer](https://en.wikipedia.org/wiki/Longplayer)

**Google Art Camera / gigapixel captures** — reverence expressed as zoom: the brushstroke made visible only to someone who leans in, never forced on the passing eye. [Digital Trends: Google's gigapixel captures](https://www.digitaltrends.com/photography/google-art-camera-gigapixel/)

## MECHANISMS

1. **Ground as atmosphere, text as the clearing.** One fixed, full-viewport gradient/canvas layer behind all typography, moving (if at all) far slower than reading speed. Turner's fog doesn't compete with the ship; it surrounds it. Content sits in a legible pocket of stillness within it.

2. **Light as navigation state.** Each section/page carries a distinct light temperature — hue, brightness, contrast of the ground layer shift on transition, like walking from gallery room to gallery room under a different skylight. Implement as CSS custom properties transitioning on route change; the *state* of the site becomes literally luminous rather than labeled.

3. **Cézanne planes as layout logic.** Replace hard dividers/grids with overlapping content blocks distinguished by subtle color-plane shift and soft depth (never a ruled line). Adjacent sections read as adjoining color patches, the way his canvases build a hillside from patch against patch.

4. **Constructive stroke as micro-texture.** A background hatching at one consistent angle (SVG `feTurbulence`/oriented line pattern, very low opacity) reads as *built*, not noisy — because the marks share a direction, the way Cézanne's do, rather than scattering randomly like static.

5. **Non-finito as a content state, not a texture.** A draft or in-progress piece is rendered visibly incomplete — trailing into raw grain, an exposed edge, an honest "unworked" zone — instead of either hiding it or fully polishing thin material to fake completeness. Incompleteness becomes information.

6. **Watercolor-layer technique for any painted surface.** Stack 20–40 near-transparent SVG shapes with jittered control points (Hobbs' method) for hero fields or dividers, instead of a two-stop CSS gradient. This is the actual mechanical difference between "looks painted" and "looks like a preset."

7. **One generative idea per surface, tuned narrow.** A flow field, particle drift, or noise field is fine — but exactly one per view, with its parameter range constrained until every output looks authored (Fidenza's discipline: wide enough to surprise, narrow enough to always cohere).

8. **Curate after generating.** Never ship the algorithm's raw first output. Generate a batch, select — Molnar's method. The site's "randomness" should feel like taste exercised on chance, not chance left unedited.

9. **Grain as varnish, applied uniformly.** A single 2–4% noise layer (`feTurbulence` + `mix-blend-mode: overlay`) across the *entire* page — text blocks, code, images alike — the way varnish unifies an old canvas, rather than one hero image getting "grungy" treatment while the rest of the site stays clinical.

10. **Zoom as intimacy, on request only.** Let a reader deliberately magnify a detail — a citation's source scan, a texture, a diagram — rather than pushing detail at them via autoplay or hover-forced reveals. Reverence is opt-in, per the Art Camera model.

11. **Durational pacing.** Any ambient/generative motion completes a cycle slower than a single reading session (minutes, not seconds) so it never reads as "playing" — only as weather that happens to be different if you return tomorrow.

12. **Chiaroscuro as hierarchy.** Dark ground, a small number of lit surfaces; what's illuminated is what matters, replacing borders and dividers with light itself. One soft light source only — never multiple competing glows — optionally nudged by cursor or scroll position, the way a spotlight, not a disco ball, follows a subject.

## THE KITSCH LINE

1. **One atmospheric idea per surface.** Fog *and* grain *and* gradient *and* parallax *and* cursor-glow stacked together is a lava lamp. Turner's atmosphere is a single unified condition of the air, not five effects competing for attention.

2. **Never animate faster than reading.** A loop or pulse under roughly eight to ten seconds reads as a screensaver. Slower than that, it reads as weather. This is the whole difference between Eno and a Windows 95 pipe animation.

3. **If it could be lifted onto any unrelated template with zero change, cut it.** Purple radial hero glow, glassmorphic card, generic AI-gold gradient — these fail not because they're colorful but because they carry no relationship to *this* content. Intentionality is the entire criterion the design-trend backlash converges on. [Creative Boom: 10 trends creatives are so over in 2026](https://www.creativeboom.com/insight/10-trends-creatives-are-so-over-in-2026/)

4. **The reader's eye should land on the words before the effect.** If atmosphere wins the first half-second of attention, contrast is miscalibrated — flip which one recedes. The room serves the sitter; it never upstages them.

5. **Non-finito must be earned, tied to real state.** A rough edge on a genuinely unfinished essay is Rodin. The same rough edge applied uniformly as a "distressed" skin over fully complete, thin content is decoration performing depth it hasn't earned — the single fastest tell of slop. And whatever atmosphere is added must sit purely in a decorative layer (CSS/SVG) over untouched semantic HTML — the light must never cost the machine-readable text beneath it.
