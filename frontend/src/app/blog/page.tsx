import Link from 'next/link'
import { formatDate, getBlogPosts } from '@/app/blog/utils'
import { PageTitle } from '@/app/components/PageTitle'

export const metadata = {
  title: 'blog',
  description: 'writing',
}

export default function Page() {
  const posts = getBlogPosts().sort((a, b) => {
    if (new Date(a.metadata.publishedAt) > new Date(b.metadata.publishedAt)) {
      return -1
    }
    return 1
  })

  return (
    <section>
      <PageTitle>blog</PageTitle>
      <ul style={{ listStyle: 'none', padding: 0, marginTop: '1rem' }}>
        {posts.map((post) => (
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
    </section>
  )
}
