import Link from 'next/link'
import { formatDate, getBlogPosts } from '@/app/blog/utils'

export function BlogPosts() {
  const allBlogs = getBlogPosts()

  return (
    <ul style={{ listStyle: 'none', padding: 0 }}>
      {allBlogs
        .sort((a, b) => {
          if (
            new Date(a.metadata.publishedAt) > new Date(b.metadata.publishedAt)
          ) {
            return -1
          }
          return 1
        })
        .slice(0, 5) // only show recent 5
        .map((post) => (
          <li key={post.slug} style={{ marginBottom: '0.75rem' }}>
            <Link 
              href={`/blog/${post.slug}`}
              style={{
                display: 'flex',
                flexDirection: 'row',
                flexWrap: 'wrap',
                gap: '0.25rem 1rem',
                alignItems: 'baseline',
              }}
            >
              <span style={{ 
                color: 'var(--color-text-muted)', 
                fontVariantNumeric: 'tabular-nums',
                fontSize: '0.9em',
                whiteSpace: 'nowrap',
              }}>
                {formatDate(post.metadata.publishedAt)}
              </span>
              <span>{post.metadata.title}</span>
            </Link>
          </li>
        ))}
    </ul>
  )
}
