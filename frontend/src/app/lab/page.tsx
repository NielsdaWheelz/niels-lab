import Link from 'next/link'
import { PageTitle } from '@/app/components/PageTitle'
import { createPageMetadata } from '@/app/site'
import styles from './lab.module.css'

export const metadata = createPageMetadata({
  title: 'Lab',
  description:
    'Mechanisms from inside a language model, rebuilt small enough to turn over by hand.',
  path: '/lab',
})

export default function Page() {
  return (
    <article>
      <PageTitle>lab</PageTitle>

      <div className="prose">
        <p>
          Mechanisms from inside a language model, rebuilt small enough to turn
          over by hand. Each one runs in your browser and calls nothing home.
        </p>
      </div>

      <ul className={`entries ${styles.list}`} role="list">
        <li className="entry">
          <Link href="/lab/sampling">how sampling works</Link>{' '}
          <time className="date" dateTime="2026-07">
            2026-07
          </time>
          <p className={styles.hook}>
            temperature, top-k, and top-p on a live token distribution — feel
            the dice reshape before the roll.
          </p>
        </li>
      </ul>
    </article>
  )
}
