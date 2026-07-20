import Link from 'next/link'
import { PageTitle } from '@/app/components/PageTitle'
import { createPageMetadata } from '@/app/site'
import { nowSections, updatedAt } from './data'
import styles from './now.module.css'

const MONTH_NAMES = [
  'jan',
  'feb',
  'mar',
  'apr',
  'may',
  'jun',
  'jul',
  'aug',
  'sep',
  'oct',
  'nov',
  'dec',
]

// pure string formatting, no Date object — updatedAt is a literal 'YYYY-MM-DD'
function formatMarginNote(dateString: string) {
  const [year, month] = dateString.split('-')
  const monthName = MONTH_NAMES[Number(month) - 1] ?? month
  return `updated ${monthName} ${year}`
}

const description =
  "What I'm doing right now: the Fractal Tech fellowship, Nexus, niels-gpt, and this site itself."

export const metadata = createPageMetadata({
  title: 'Now',
  description,
  path: '/now',
})

export default function NowPage() {
  const marginNote = formatMarginNote(updatedAt)

  return (
    <section className={styles.page}>
      <p className="home-kicker">
        <span className="status-dot" aria-hidden="true" />
        dated, not evergreen
      </p>
      <PageTitle>now</PageTitle>
      <p className={`margin-note ${styles.marginNote}`}>{marginNote}</p>
      <p className="page-intro">
        No history, no highlight reel — just the present tense. For the history,
        there&apos;s <Link href="/cv">the CV</Link>.
      </p>

      <div className={styles.sections}>
        {nowSections.map((section) => (
          <section
            key={section.id}
            className={styles.section}
            aria-labelledby={`${section.id}-heading`}
          >
            <header className={styles.sectionHeader}>
              <p className={styles.sectionIndex}>{section.index}</p>
              <h2
                id={`${section.id}-heading`}
                className={styles.sectionHeading}
              >
                {section.heading}
              </h2>
            </header>
            <ul className={styles.entryList}>
              {section.entries.map((entry) => (
                <li key={entry.title} className={styles.entry}>
                  <p className={styles.entryTitle}>{entry.title}</p>
                  <p className={styles.entryDetail}>
                    {entry.detail}
                    {entry.href ? (
                      <>
                        {' '}
                        {entry.href.startsWith('/') ? (
                          <Link href={entry.href}>
                            {entry.linkLabel ?? entry.href}
                          </Link>
                        ) : (
                          <a
                            href={entry.href}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            {entry.linkLabel ?? entry.href}
                          </a>
                        )}
                      </>
                    ) : null}
                  </p>
                </li>
              ))}
            </ul>
            {section.id === 'open-to' ? (
              <div className={styles.sectionAction}>
                <a
                  href="mailto:niels.erik.nandal@gmail.com"
                  className="hero-action hero-action-secondary"
                >
                  email me <span aria-hidden="true">↗</span>
                </a>
              </div>
            ) : null}
          </section>
        ))}
      </div>

      <p className={styles.footer}>
        This is a{' '}
        <a
          href="https://nownownow.com/about"
          target="_blank"
          rel="noopener noreferrer"
        >
          now page
        </a>
        : a dated snapshot instead of an evergreen bio. If it looks stale,
        that&apos;s a bug — email me.
      </p>
    </section>
  )
}
