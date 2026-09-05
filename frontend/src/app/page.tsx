import type { Metadata } from 'next'
import Link from 'next/link'
import { PageTitle } from '@/app/components/PageTitle'
import { ListBody } from '@/app/components/ListBody'
import { lastWritten, lists } from '@/content/lists'
import { band } from '@/lib/hour'
import { hourTable } from '@/content/hours'
import { bio, bookTitle, createPageMetadata, siteDescription } from '@/app/site'
import styles from './home.module.css'

export const metadata: Metadata = {
  ...createPageMetadata({
    title: bookTitle,
    description: siteDescription,
    path: '/',
  }),
  // The book names itself; the layout's title template must not re-suffix it.
  title: { absolute: bookTitle },
}

// The hour epigraph and the default-open list both follow the real SF hour
// band (spec §4, §6) — static-first, re-derived hourly, never client clock.
export const revalidate = 3600

export default function Page() {
  const { epigraph, opens } = hourTable[band(new Date())]
  // validate-content.mjs proves every HourTable epigraph resolves to a
  // published list and an in-range entry (spec §6) before the build can
  // succeed, so this lookup cannot miss at runtime.
  const epigraphList = lists.find((list) => list.slug === epigraph.list)!
  const epigraphText = epigraphList.entries[epigraph.index].text

  return (
    <div className="canon leaf">
      <header>
        <PageTitle>{bookTitle}</PageTitle>
        <p className={styles.subline}>
          Lists, kept and dated. A claim links to its evidence or goes as plain
          observation.
        </p>
        <p className={styles.subline}>{bio}</p>
      </header>
      <hr />
      <p className={styles.epigraph}>{epigraphText}</p>
      <section aria-label="The lists">
        {lists.map((list) => {
          const last = lastWritten(list)
          return (
            <details
              key={list.slug}
              id={list.slug}
              className={styles.list}
              open={list.slug === opens}
            >
              <summary className={styles.row}>
                <h2 className="list-title">{list.title}</h2>
                <span className={`gloss ${styles.rowMeta}`}>
                  <span className="count">{list.entries.length}</span>
                  {' · '}
                  <time className="date" dateTime={last}>
                    {last}
                  </time>
                </span>
              </summary>
              <ListBody list={list} />
              <p className={`chrome ${styles.permalink}`}>
                <Link
                  href={`/lists/${list.slug}`}
                  aria-label={`${list.title} — permalink`}
                >
                  § permalink
                </Link>
              </p>
            </details>
          )
        })}
      </section>
    </div>
  )
}
