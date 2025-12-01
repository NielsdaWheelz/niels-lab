import Link from 'next/link'
import { formatDate, getBraindumps } from '@/app/braindumps/utils'

export function BraindumpsList() {
  const allDumps = getBraindumps()

  return (
    <div>
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
          <Link
            key={dump.slug}
            className="flex flex-col space-y-1 mb-4"
            href={`/braindumps/${dump.slug}`}
          >
            <div className="w-full flex flex-col md:flex-row space-x-0 md:space-x-2">
              <p className="text-neutral-600 dark:text-neutral-400 w-[100px] tabular-nums">
                {formatDate(dump.metadata.publishedAt, false)}
              </p>
              <p className="text-neutral-900 dark:text-neutral-100 tracking-tight">
                {dump.metadata.title}
              </p>
            </div>
          </Link>
        ))}
    </div>
  )
}

