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
          <li key={post.slug} style={{ marginBottom: '0.5rem' }}>
            <Link href={`/blog/${post.slug}`}>
              <span style={{ color: '#666', marginRight: '1rem', fontVariantNumeric: 'tabular-nums' }}>
                {formatDate(post.metadata.publishedAt)}
              </span>
              {post.metadata.title}
            </Link>
          </li>
        ))}
    </ul>
  )
}
