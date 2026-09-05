import Link from 'next/link'
import { PageTitle } from '@/app/components/PageTitle'
import { createPageMetadata } from '@/app/site'
import styles from './lab.module.css'

export const metadata = createPageMetadata({
  title: 'Lab',
  description:
    'Interactive pages that demonstrate language-model mechanisms, starting with token sampling.',
  path: '/lab',
})

export default function Page() {
  return (
    <article className="canon leaf">
      <PageTitle>lab</PageTitle>

      <div className="prose">
        <p>
          Mechanisms from inside a language model, rebuilt small enough to
          operate by hand. Each one runs in your browser and makes no network
          requests.
        </p>
      </div>

      <ul className={`entries ${styles.list}`} role="list">
        <li className="entry">
          <Link href="/lab/sampling">how sampling works</Link>{' '}
          <time className="date" dateTime="2026-07">
            2026-07
          </time>
          <p className={styles.hook}>
            one toy distribution, three knobs, and the discarded mass kept
            visible.
          </p>
        </li>
      </ul>
    </article>
  )
}
