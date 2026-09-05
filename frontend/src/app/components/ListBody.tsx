import Link from 'next/link'
import type { Evidence, PillowList } from '@/content/lists'
import styles from './ListBody.module.css'

// Evidence body: the label is the link to the primary source (spec §4:
// the summary is the celadon mark; opening yields label + link).
function Source({ evidence }: { evidence: Evidence }) {
  if ('kind' in evidence) {
    return (
      // eslint-disable-next-line @next/next/no-img-element -- local static asset, dimensions unknown to the schema
      <img src={evidence.src} alt={evidence.label} loading="lazy" />
    )
  }
  if (evidence.href.startsWith('/')) {
    return <Link href={evidence.href}>{evidence.label}</Link>
  }
  return <a href={evidence.href}>{evidence.label}</a>
}

export function ListBody({ list }: { list: PillowList }) {
  return (
    <>
      {list.note ? <p className={styles.note}>{list.note}</p> : null}
      <ul className={`entries ${styles.body}`} role="list">
        {list.entries.map((entry, index) => (
          <li className={`entry ${styles.entry}`} key={index}>
            {entry.text}
            {entry.evidence ? (
              <details className={`evidence ${styles.evidence}`}>
                {/* The visible summary is the › mark alone (CSS); the label
                    names the disclosure for assistive technology. On wide
                    viewports the details is superseded by the
                    always-visible margin aside below (spec §8 kill list)
                    and the summary is dropped. */}
                <summary>
                  <span className="sr-only">{entry.evidence.label}</span>
                </summary>
                <aside className={`gloss ${styles.source}`}>
                  <Source evidence={entry.evidence} />
                </aside>
              </details>
            ) : null}
          </li>
        ))}
      </ul>
    </>
  )
}
