import Link from 'next/link'
import { PageTitle } from '@/app/components/PageTitle'
import { createPageMetadata } from '@/app/site'
import { lists } from '@/content/lists'
import { hourTable } from '@/content/hours'
import type { HourBand } from '@/lib/hour'
import { AGE_STEPS, FAILURE_FACTOR } from '@/lib/ledger/inkAge'
import styles from './colophon.module.css'

export const metadata = createPageMetadata({
  title: 'Colophon',
  description:
    'How this site is set: the canon grid, the six tokens, the hour bands, the ledger’s ink-age law, and the two client-side confessions.',
  path: '/colophon',
})

// The whole palette (globals.css). Chips are painted with the live custom
// property, never a frozen value, so the theme toggle repaints them.
const tokens = [
  { name: '--ink', role: 'the page ground', chip: styles.ink },
  { name: '--lamplight', role: 'body text', chip: styles.lamplight },
  { name: '--murasaki', role: 'headings and links', chip: styles.murasaki },
  {
    name: '--celadon',
    role: 'evidence marks, dates, counts, lessons',
    chip: styles.celadon,
  },
  { name: '--rule', role: 'hairlines', chip: styles.rule },
  {
    name: '--faded',
    role: 'secondary text; the ink-age floor',
    chip: styles.faded,
  },
]

// Five more chips off the same palette, one per ledger ink-age step. Labels
// are computed from AGE_STEPS, not typed by hand, so they cannot drift from
// the exported constant.
const ageChipClasses = [
  styles.age0,
  styles.age1,
  styles.age2,
  styles.age3,
  styles.age4,
]
const ageTokens = AGE_STEPS.map((threshold, index) => {
  const next = AGE_STEPS[index + 1]
  const role =
    next === undefined
      ? `${threshold} days or older`
      : `${threshold}–${next - 1} days`
  return { name: `--age-${index}`, role, chip: ageChipClasses[index] }
})

// The six hour bands, America/Los_Angeles wall clock (src/lib/hour.ts).
// Ranges are fixed by spec §6 and asserted by hour.test.ts; listed here for
// display order, not recomputed.
const bands: { band: HourBand; range: string }[] = [
  { band: 'small-hours', range: '00:00–05:00' },
  { band: 'dawn', range: '05:00–08:00' },
  { band: 'morning', range: '08:00–12:00' },
  { band: 'afternoon', range: '12:00–17:00' },
  { band: 'dusk', range: '17:00–20:00' },
  { band: 'evening', range: '20:00–24:00' },
]

// validate-content.mjs fails the build if a band's list or index does not
// resolve, so this lookup cannot miss once the page is serving traffic.
function listBySlug(slug: string) {
  const list = lists.find((candidate) => candidate.slug === slug)
  if (!list) throw new Error(`colophon: no published list "${slug}"`)
  return list
}

export default function ColophonPage() {
  return (
    <article className="canon leaf">
      <PageTitle>colophon</PageTitle>

      <div className="prose">
        <h2>What this is</h2>
        <p>
          A zuihitsu (“following the brush”) after Sei Shōnagon, who kept lists
          in a drawer of her writing desk around the year 1000 and was
          embarrassed when they circulated. Hers were merciless and exact:
          elegant things, hateful things, things that quicken the heart. This
          site borrows the form with one amendment a Heian court lady never
          needed: every checkable claim is wired to primary evidence, and what
          cannot be checked is written as plain observation.
        </p>

        <h2>How it is made</h2>
        <p>
          Next.js, App Router, React Server Components; pages are static first
          and assembled on the server. The lists are not a database and not
          markdown — they are typed TypeScript data, reviewed like code, and the
          type checker is the CMS. The CSS is written by hand: one file of
          tokens, a module per component, no framework. Two faces do all the
          work: Newsreader for everything you read, iA Writer Quattro for dates,
          counts, and chrome. The tokens are named for what they are — ink for
          the ground, lamplight for the text, murasaki for headings and links,
          celadon for evidence and dates.
        </p>

        <dl className={styles.rows}>
          <div className={styles.row}>
            <dt className="chrome">Newsreader</dt>
            <dd className={styles.reading}>Things that quicken the heart</dd>
          </div>
          <div className={styles.row}>
            <dt className="chrome">iA Writer Quattro</dt>
            <dd className="date">2026-08-09 · 250ms · 68ch</dd>
          </div>
        </dl>

        <dl className={styles.rows}>
          {tokens.map((token) => (
            <div className={styles.row} key={token.name}>
              <dt className="chrome">
                <span
                  className={`${styles.chip} ${token.chip}`}
                  aria-hidden="true"
                />
                {token.name}
              </dt>
              <dd className={styles.role}>{token.role}</dd>
            </div>
          ))}
        </dl>

        <p>
          iA Writer Quattro is vendored in the repository under the SIL Open
          Font License, alongside the Newsreader files the OG card is set in;
          the reading face itself is fetched once at build time and self-hosted.
          The chips above are painted with the live custom properties, so the
          theme toggle repaints them. Motion is budgeted at one animation, a
          250ms ink-reveal when a disclosure opens; nothing else on the site
          moves.
        </p>

        <h2>The canon grid</h2>
        <p>
          Every leaf renders on one CSS grid, four tracks, fixed once in{' '}
          <code>globals.css</code>: a 2.5rem inner gutter; the text column at{' '}
          <code>--measure</code>, 68ch, unchanged since the first draft; an
          outer margin at round(0.4 × <code>--measure</code>), 27ch — exactly
          one Quattro gloss line, since an entry carries at most one; and a
          spare <code>1fr</code> track that takes whatever width is left. Below
          76rem the grid collapses to a single block column, and the outer
          margin folds into each entry’s own disclosure.
        </p>

        <h2>The hour</h2>
        <p>
          The index reads the San Francisco clock, server-side.{' '}
          <code>src/lib/hour.ts</code> maps the current instant to one of six
          America/Los_Angeles wall-clock bands;{' '}
          <code>src/content/hours.ts</code> pairs each band with a corpus entry
          to quote as the epigraph, and a list to hold open, referenced by
          (list, index) rather than retyped — so the quote is byte-identical to
          its source by construction. The page revalidates hourly; the band is
          computed from the server clock at every render.
        </p>

        <div className={styles.hoursWrap}>
          <table className={styles.hours}>
            <caption className="sr-only">
              The six hour bands, each with its epigraph and the list it holds
              open.
            </caption>
            <thead>
              <tr>
                <th scope="col">Band</th>
                <th scope="col">Hour, SF</th>
                <th scope="col">Epigraph</th>
                <th scope="col">Opens</th>
              </tr>
            </thead>
            <tbody>
              {bands.map(({ band, range }) => {
                const row = hourTable[band]
                const epigraphList = listBySlug(row.epigraph.list)
                const epigraphEntry = epigraphList.entries[row.epigraph.index]
                const opensList = listBySlug(row.opens)
                return (
                  <tr key={band}>
                    <th scope="row" className="chrome">
                      {band}
                    </th>
                    <td className="chrome">{range}</td>
                    <td className={styles.epigraph}>
                      “{epigraphEntry.text}”
                      <Link
                        href={`/lists/${epigraphList.slug}`}
                        className={`chrome ${styles.epigraphSource}`}
                      >
                        {epigraphList.title}
                      </Link>
                    </td>
                    <td>
                      <Link
                        href={`/lists/${opensList.slug}`}
                        className="chrome"
                      >
                        {opensList.title}
                      </Link>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        <h2>The ledger’s ink-age</h2>
        <p>
          <Link href="/log">The log</Link> dries with age.{' '}
          <code>src/lib/ledger/inkAge.ts</code> steps a row’s age in days
          against <code>AGE_STEPS</code> ({AGE_STEPS.join(', ')}) and maps the
          step to one of five tokens, <code>--age-0</code> through{' '}
          <code>--age-4</code>, each interpolated from <code>--lamplight</code>{' '}
          toward <code>--faded</code> and checked AA on its own ground. A struck
          failure’s age is multiplied by <code>FAILURE_FACTOR</code> (
          {FAILURE_FACTOR}) before that lookup, so it dries twice as fast as an
          on-time neighbor. Its lesson is exempt: full celadon at any age,
          because the lesson is the part meant to survive.{' '}
          <code>prefers-contrast: more</code> collapses every step back to full
          ink.
        </p>

        <dl className={styles.rows}>
          {ageTokens.map((token) => (
            <div className={styles.row} key={token.name}>
              <dt className="chrome">
                <span
                  className={`${styles.chip} ${token.chip}`}
                  aria-hidden="true"
                />
                {token.name}
              </dt>
              <dd className={styles.role}>{token.role}</dd>
            </div>
          ))}
        </dl>

        <h2>The theme flip</h2>
        <p>
          Light is canonical now (the client’s word: “paper,” 2026-09-05); dark
          is not its inversion. On wide viewports the dark theme paints the text
          column a second, flatter tone against the deep ground — two flat
          colors and grid geometry, no gradient. <code>theme.ts</code>’s default
          flipped to match; the toggle and its storage key are unchanged.
        </p>

        <h2>The confession</h2>
        <p>
          Two client components run on this site: the theme toggle, and the
          sampling playground in <Link href="/lab">the lab</Link>, which rolls
          its dice in your browser or not at all. Everything else is markup sent
          finished from the server. Evidence discloses as <code>details</code>{' '}
          elements; they open without JavaScript, in reader mode, and under
          curl. If a sentence does not exist in the initial HTML, it does not
          exist.
        </p>

        <h2>Evidence</h2>
        <p>
          Entries that make a checkable claim carry a small celadon mark. Open
          it and you get a label and a link to the primary source — a
          repository, a checkpoint, a live system, a page of working notes.
          Entries without proof ship unmarked; there are no placeholders, and a
          dead evidence link fails the build. The lifting numbers are the
          deliberate exception: they appear as written observation only, the way
          a diary would have them.
        </p>

        <h2>Homunculus, forthcoming</h2>
        <p>
          Version two of this site will contain the Homunculus: a small language
          model trained on my own writing, answering questions as the site
          rather than about it. Pessoa kept heteronyms; I intend to keep one
          made of matrices. It is not built, and this paragraph is currently its
          entire implementation.
        </p>

        <h2>Provenance</h2>
        <p>
          Designed and written by Niels-Erik Nandal, built in collaboration with
          Claude, Anthropic’s model, working as a pair programmer. The git
          history records who did what, commit by commit; nothing here pretends
          otherwise.
        </p>
        <p className={styles.signoff}>Nandal faciebat, San Francisco.</p>
      </div>
    </article>
  )
}
