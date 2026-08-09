import Link from 'next/link'
import { PageTitle } from '@/app/components/PageTitle'
import { createPageMetadata } from '@/app/site'
import styles from './colophon.module.css'

export const metadata = createPageMetadata({
  title: 'Colophon',
  description:
    'How this site is set: the six tokens, the two faces, the client-JS confession, and the Homunculus plan.',
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
    role: 'evidence marks, dates, counts',
    chip: styles.celadon,
  },
  { name: '--rule', role: 'hairlines', chip: styles.rule },
  { name: '--faded', role: 'secondary text', chip: styles.faded },
]

export default function ColophonPage() {
  return (
    <article>
      <PageTitle>colophon</PageTitle>

      <div className="prose">
        <h2>What this is</h2>
        <p>
          A zuihitsu — &quot;following the brush&quot; — after Sei Shōnagon, who
          kept lists in a drawer of her writing desk around the year 1000 and
          was embarrassed when they circulated. Hers were merciless and exact:
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
          celadon for evidence and dates. Dark is the canonical theme; this is
          night writing. Light is its true inverse, not an afterthought.
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

        <h2>The confession</h2>
        <p>
          Two client components run on this site: the theme toggle, and the
          sampling playground in <Link href="/lab">the lab</Link>, which rolls
          its dice in your browser or not at all. Everything else is markup sent
          finished from the server. Evidence disclosures are details elements;
          they open without JavaScript, in reader mode, and under curl. If a
          sentence does not exist in the initial HTML, it does not exist.
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
          Claude, Anthropic&apos;s model, working as a pair programmer. The git
          history records who did what, commit by commit; nothing here pretends
          otherwise.
        </p>
      </div>
    </article>
  )
}
