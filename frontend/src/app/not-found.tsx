import type { Metadata } from 'next'
import Link from 'next/link'

export const metadata: Metadata = {
  title: 'Things that are not where you left them',
}

export default function NotFound() {
  return (
    <section className="canon leaf">
      <h1 className="list-title">Things that are not where you left them</h1>
      <hr />
      <ul className="entries" role="list">
        <li className="entry">
          The puck, after the deke. The goaltender is looking for it too.
        </li>
        <li className="entry">
          A bar re-racked by a stranger, ten pounds lighter than you remember
          leaving it.
        </li>
        <li className="entry">
          Your place in the book. The bookmark fell out somewhere over the
          Atlantic.
        </li>
        <li className="entry">
          This page. Renamed, moved, or never written; the server declines to
          speculate.
        </li>
        <li className="entry">
          <Link href="/">The index, which has not moved.</Link>
        </li>
      </ul>
    </section>
  )
}
