import Link from 'next/link'
import type { Evidence, PillowList } from '@/content/lists'
import styles from './ListBody.module.css'

// Evidence body: the primary source itself — its address as the link text.
function Source({ evidence }: { evidence: Evidence }) {
  if ('kind' in evidence) {
    return (
      // eslint-disable-next-line @next/next/no-img-element -- local static asset, dimensions unknown to the schema
      <img src={evidence.src} alt={evidence.label} loading="lazy" />
    )
  }
  if (evidence.href.startsWith('/')) {
    return <Link href={evidence.href}>{evidence.href}</Link>
  }
  return <a href={evidence.href}>{evidence.href}</a>
}

export function ListBody({ list }: { list: PillowList }) {
  return (
    <>
      {list.note ? <p className={styles.note}>{list.note}</p> : null}
      <ul className={`entries ${styles.body}`}>
        {list.entries.map((entry, index) => (
          <li className="entry" key={index}>
            {entry.text}
            {entry.evidence ? (
              <details className="evidence">
                <summary>{entry.evidence.label}</summary>
                <p className={`chrome ${styles.source}`}>
                  <Source evidence={entry.evidence} />
                </p>
              </details>
            ) : null}
          </li>
        ))}
      </ul>
    </>
  )
}
