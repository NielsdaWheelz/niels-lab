import Link from 'next/link'
import { formatDate, getWritingPosts } from '@/app/writing/utils'
import { PageTitle } from '@/app/components/PageTitle'

export const metadata = {
  title: 'writing',
  description: 'Technical notes and build writeups',
}

export default function Page() {
  const posts = getWritingPosts().sort((a, b) => {
    if (new Date(a.metadata.publishedAt) > new Date(b.metadata.publishedAt)) {
      return -1
    }
    return 1
  })

  return (
    <section>
      <PageTitle>writing</PageTitle>
      <p className="page-intro">
        Notes on model-building, engineering practice, and what I learned while
        shipping things that were hard enough to matter.
      </p>
      <ul className="writing-list">
        {posts.map((post) => (
          <li key={post.slug} className="writing-item">
            <Link href={`/writing/${post.slug}`} className="writing-title">
              {post.metadata.title}
            </Link>
            <p className="writing-meta">
              {formatDate(post.metadata.publishedAt)}
            </p>
            <p className="writing-summary">{post.metadata.summary}</p>
          </li>
        ))}
      </ul>
    </section>
  )
}
