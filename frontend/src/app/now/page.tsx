import Link from 'next/link'
import { PageTitle } from '@/app/components/PageTitle'
import { createPageMetadata } from '@/app/site'
import styles from './now.module.css'

const description =
  'A dated diary entry: Solid, Braudel, the meet total, and what this site became in August.'

export const metadata = createPageMetadata({
  title: 'Now',
  description,
  path: '/now',
})

export default function NowPage() {
  return (
    <article className="canon leaf">
      <PageTitle>now</PageTitle>
      <p className={`date ${styles.dateline}`}>
        <time dateTime="2026-08-09">August 9, 2026</time> · San Francisco
      </p>

      <div className="prose">
        <p>
          Days at Solid, building infrastructure for always-on agents. I have
          been a senior engineer there since March. The work is the good kind of
          unglamorous: the parts that must hold when nobody is watching.
        </p>
        <p>
          Braudel&apos;s <em>Mediterranean</em> in the mornings, forty pages at
          a sitting; most of the book is still ahead.
        </p>
        <p>
          The meet total stands at 1,055 — squat 415, bench 275, deadlift 365.
          Training is ordinary on purpose: heavy triples, long warmups, the same
          empty-bar start every session. Soccer most evenings, where I defend.
        </p>
        <p>
          In August this site became a book of lists. The{' '}
          <Link href="/colophon">colophon</Link> explains how; the{' '}
          <Link href="/log">log</Link> shows whether.
        </p>
      </div>

      <hr />
      <p className={styles.note}>
        Written by hand, dated, replaced when false.
      </p>
    </article>
  )
}
