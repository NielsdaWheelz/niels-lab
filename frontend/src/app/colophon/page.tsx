import { PageTitle } from '@/app/components/PageTitle'
import { createPageMetadata } from '@/app/site'
import { ColorSwatch } from './ColorSwatch'
import styles from './colophon.module.css'

export const metadata = createPageMetadata({
  title: 'Colophon',
  description:
    "The design tokens, type specimens, stack, and provenance behind this site's sketchbook — the notebook inspecting itself.",
  path: '/colophon',
})

const paletteTokens = [
  {
    varName: '--color-bg',
    role: 'the page — cream paper by day, midnight blueprint by night',
    light: '#fbf4e2',
    dark: '#10151c',
  },
  {
    varName: '--color-bg-alt',
    role: 'raised panel — cards, code blocks, callouts',
    light: '#f4ebd3',
    dark: '#171e27',
  },
  {
    varName: '--color-text',
    role: 'ink — body copy',
    light: '#2e2a23',
    dark: '#e6dfcc',
  },
  {
    varName: '--color-text-muted',
    role: 'faded ink — captions, meta, secondary copy',
    light: '#6f675a',
    dark: '#97917f',
  },
  {
    varName: '--color-terracotta',
    role: 'primary accent — links, CTAs, the stamp',
    light: '#d4552a',
    dark: '#ef8250',
  },
  {
    varName: '--color-sage',
    role: 'secondary accent — the deep teal, hovers, marginalia',
    light: '#17697a',
    dark: '#5cb8c9',
  },
  {
    varName: '--color-gold',
    role: 'tertiary accent — the ochre, used sparingly',
    light: '#c0932f',
    dark: '#d9b35e',
  },
  {
    varName: '--color-border',
    role: 'paper edge — dividers, card outlines',
    light: '#e4d9bf',
    dark: '#29323e',
  },
] as const

const stack = [
  {
    name: 'Next.js 16',
    note: 'App Router, React Server Components — the whole site runs on it.',
  },
  {
    name: 'React 19',
    note: 'actions and refs-as-props mean fewer wrappers between intent and pixels.',
  },
  {
    name: 'TypeScript, strict',
    note: '"readable is a feature" applies to the source too, not just the interface.',
  },
  {
    name: 'Bun',
    note: 'package manager and script runner for install, dev, and build.',
  },
  {
    name: 'next-mdx-remote',
    note: 'writing lives as MDX in /writing and renders server-side — no client bundle for prose.',
  },
  {
    name: 'rough.js',
    note: 'the hand-drawn canvas accents: the 404 map, the CV timeline, the homepage system diagram.',
  },
  {
    name: 'sugar-high',
    note: "syntax highlighting, retokenized into the site's own warm palette.",
  },
  {
    name: 'hand-rolled CSS',
    note: 'one globals.css plus a CSS module per component — no Tailwind, no UI kit.',
  },
  {
    name: 'Vercel',
    note: 'build, deploy, analytics, and speed insights.',
  },
] as const

const ingredients = [
  'graph-paper ruling: two layered linear-gradients on <html>, 32px apart, tinted with --color-ruling',
  'a custom dot cursor: an inline SVG data URI, terracotta over text, teal over links',
  'a ctrl+k terminal for poking at the site from the keyboard instead of the mouse',
  'dark mode as "midnight blueprint" — one data-theme attribute flips every token on this page, no second stylesheet',
  'hand-drawn SVG and canvas annotations: the circled "receipts" on the homepage, the torn-page 404 scene, the CV timeline dots',
]

export default function ColophonPage() {
  return (
    <section className={styles.page}>
      <PageTitle>colophon</PageTitle>
      <p className="page-intro">
        A colophon used to be the printer&rsquo;s mark at the back of a book —
        who set the type, on what press, in which face. This is that page,
        except the book is a website and it can show its work instead of just
        describing it.
      </p>
      <p className={styles.invite}>
        Everything below is rendered with the same custom properties, fonts, and
        libraries that draw this very sentence. Toggle the theme in the nav —
        the swatches, and the whole page around them, will follow.
      </p>

      <section className={styles.section} aria-labelledby="palette-heading">
        <header className="section-header">
          <p className="section-index">01 / palette, live</p>
          <h2 id="palette-heading">Eight variables, two moods.</h2>
          <p>
            Every chip below is painted with the actual custom property, not a
            frozen hex — flip the theme and watch them repaint. Click one to
            copy its variable name.
          </p>
        </header>

        <div className={styles.swatchGrid}>
          {paletteTokens.map((token) => (
            <ColorSwatch key={token.varName} {...token} />
          ))}
        </div>

        <p className={`margin-note ${styles.marginAside}`}>
          yes — I know you might be reading these values in devtools right now.
          that&rsquo;s the point.
        </p>

        <div className={styles.atmosphereGrid}>
          <div className={styles.atmosphereCard}>
            <div className={styles.rulingDemo} aria-hidden="true" />
            <p>
              <code>--color-ruling</code> — the graph-paper grid, a teal at
              roughly 5% opacity so it reads as texture, not noise.
            </p>
          </div>
          <div className={styles.atmosphereCard}>
            <div className={styles.shadowDemo} aria-hidden="true" />
            <p>
              <code>--color-shadow</code> — the one shadow color on the site,
              reused everywhere something needs to lift off the paper.
            </p>
          </div>
        </div>
      </section>

      <section className={styles.section} aria-labelledby="type-heading">
        <header className="section-header">
          <p className="section-index">02 / type specimens</p>
          <h2 id="type-heading">Three faces, three jobs.</h2>
        </header>

        <div className={styles.specimen}>
          <p className={styles.specimenLabel}>Newsreader — display serif</p>
          <p className={styles.specimenSerif}>
            Build the magic, keep the receipts.
          </p>
          <p className={styles.specimenNote}>
            Headlines, page titles, the occasional italic aside.
          </p>
        </div>

        <div className={styles.specimen}>
          <p className={styles.specimenLabel}>Geist Mono — the workhorse</p>
          <p className={styles.specimenMono}>
            $ bun run dev · NAV · 03:14:07 · 2/2 checks passed
          </p>
          <p className={styles.specimenNote}>
            Navigation, timestamps, labels, and every code block on the site.
          </p>
        </div>

        <div className={styles.specimen}>
          <p className={styles.specimenLabel}>Caveat — the marginalia</p>
          <p className={styles.specimenHand}>
            a note scribbled in the margin, because rigor still needs room to
            breathe
          </p>
          <p className={styles.specimenNote}>
            Asides and annotations — the handwriting layer over the grid.
          </p>
        </div>
      </section>

      <section className={styles.section} aria-labelledby="stack-heading">
        <header className="section-header">
          <p className="section-index">03 / stack</p>
          <h2 id="stack-heading">What it&rsquo;s actually made of.</h2>
        </header>
        <dl className={styles.stackList}>
          {stack.map((item) => (
            <div className={styles.stackItem} key={item.name}>
              <dt>{item.name}</dt>
              <dd>{item.note}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className={styles.section} aria-labelledby="ingredients-heading">
        <header className="section-header">
          <p className="section-index">04 / ingredients</p>
          <h2 id="ingredients-heading">The small decisions that add up.</h2>
        </header>
        <ul className={styles.ingredientsList}>
          {ingredients.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </section>

      <section className={styles.section} aria-labelledby="provenance-heading">
        <header className="section-header">
          <p className="section-index">05 / provenance</p>
          <h2 id="provenance-heading">Who built this, honestly.</h2>
        </header>
        <p>
          This site is designed and built by Niels, in collaboration with Claude
          — Anthropic&rsquo;s model, working as a pair-programmer with strong
          opinions about CSS. That is not a caveat tucked at the bottom of the
          page; it is the point. The same &ldquo;build the magic, keep the
          receipts&rdquo; ethic that runs through the rest of this site applies
          to how the site itself got made.
        </p>
        <p>The git history keeps the receipts, commit by commit:</p>
        <pre>
          <code>Co-Authored-By: Claude &lt;noreply@anthropic.com&gt;</code>
        </pre>
        <p>
          No ghostwriting, no pretending otherwise. If a line of CSS turned out
          clever, it is worth knowing whose idea it was.
        </p>
      </section>
    </section>
  )
}
