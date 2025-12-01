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
          <li key={dump.slug} style={{ marginBottom: '0.75rem' }}>
            <Link 
              href={`/braindumps/${dump.slug}`}
              style={{
                display: 'flex',
                flexDirection: 'row',
                flexWrap: 'wrap',
                gap: '0.25rem 1rem',
                alignItems: 'baseline',
              }}
            >
              <span style={{ 
                color: '#666', 
                fontVariantNumeric: 'tabular-nums',
                fontSize: '0.9em',
                whiteSpace: 'nowrap',
              }}>
                {formatDate(dump.metadata.publishedAt)}
              </span>
              <span>{dump.metadata.title}</span>
            </Link>
          </li>
        ))}
    </ul>
  )
}
