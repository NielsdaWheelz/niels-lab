import { PageTitle } from '@/app/components/PageTitle'
import { createPageMetadata } from '@/app/site'
import { getLedger, isStale } from '@/lib/ledger'
import styles from './log.module.css'

export const revalidate = 86400

const description =
  'A dated ledger of lifts, pages, and shipped work; failed entries stay, struck through, with the lesson attached.'

export const metadata = createPageMetadata({
  title: 'Log',
  description,
  path: '/log',
})

export default async function LogPage() {
  const events = await getLedger()
  const today = new Date().toISOString().slice(0, 10)

  return (
    <section>
      <PageTitle>log</PageTitle>
      <p className="prose">
        Training, reading, shipping. The failures stay in.
      </p>

      {isStale(events, today) ? (
        <p className={`chrome ${styles.deload}`}>
          Nothing logged in the last fortnight: no row written by hand, no push
          to a public main.
        </p>
      ) : null}

      <ol className={styles.rows} role="list">
        {events.map((event) => {
          const text = event.failed ? <s>{event.text}</s> : event.text

          return (
            <li key={`${event.date} ${event.text}`} className={styles.row}>
              <time className="date" dateTime={event.date}>
                {event.date}
              </time>
              <span className={styles.kind}>{event.kind}</span>
              <span>
                {event.href ? <a href={event.href}>{text}</a> : text}
                {event.failed ? (
                  <span className={styles.lesson}>
                    {' — '}
                    <span className="sr-only">failed: </span>
                    {event.failed.lesson}
                  </span>
                ) : null}
              </span>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
