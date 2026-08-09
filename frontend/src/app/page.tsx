import type { Metadata } from 'next'
import Link from 'next/link'
import { PageTitle } from '@/app/components/PageTitle'
import { ListBody } from '@/app/components/ListBody'
import { lastWritten, lists } from '@/content/lists'
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

export default function Page() {
  return (
    <>
      <header>
        <PageTitle>{bookTitle}</PageTitle>
        <p className={styles.subline}>
          Lists, kept and dated. A claim links to its evidence or goes as plain
          observation.
        </p>
        <p className={styles.subline}>{bio}</p>
      </header>
      <hr />
      <section aria-label="The lists">
        {lists.map((list, index) => {
          const last = lastWritten(list)
          return (
            <details
              key={list.slug}
              id={list.slug}
              className={styles.list}
              open={index === 0}
            >
              <summary className={styles.row}>
                <h2 className="list-title">{list.title}</h2>
                <span className={`chrome ${styles.rowMeta}`}>
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
    </>
  )
}
