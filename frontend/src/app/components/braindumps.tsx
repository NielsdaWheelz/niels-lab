import Link from 'next/link'
import { formatDate, getBraindumps } from '@/app/braindumps/utils'

export function BraindumpsList() {
  const allDumps = getBraindumps()

  return (
    <ul style={{ listStyle: 'none', padding: 0 }}>
      {allDumps
        .sort((a, b) => {
          if (
            new Date(a.metadata.publishedAt) > new Date(b.metadata.publishedAt)
          ) {
            return -1
          }
          return 1
        })
        .map((dump) => (
          <li key={dump.slug} style={{ marginBottom: '0.5rem' }}>
            <Link href={`/braindumps/${dump.slug}`}>
              <span style={{ color: '#666', marginRight: '1rem', fontVariantNumeric: 'tabular-nums' }}>
                {formatDate(dump.metadata.publishedAt)}
              </span>
              {dump.metadata.title}
            </Link>
          </li>
        ))}
    </ul>
  )
}
