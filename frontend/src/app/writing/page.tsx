import Link from 'next/link'
import { getWritingPosts } from '@/lib/content'
import { PageTitle } from '@/app/components/PageTitle'
import { createPageMetadata } from '@/app/site'
import styles from '@/app/posts.module.css'

export const metadata = createPageMetadata({
  title: 'Writing',
  description:
    'Technical notes on machine learning, AI systems, and software engineering by Niels Erik Nandal.',
  path: '/writing',
})

export default function Page() {
  return (
    <section>
      <PageTitle>writing</PageTitle>
      <ul className={styles.index}>
        {getWritingPosts().map((post) => (
          <li key={post.slug} className={styles.row}>
            <div className={styles.head}>
              <h2 className="list-title">
                <Link href={`/writing/${post.slug}`}>
                  {post.metadata.title}
                </Link>
              </h2>
              <time className="date" dateTime={post.metadata.publishedAt}>
                {post.metadata.publishedAt}
              </time>
            </div>
            <p className={styles.summary}>{post.metadata.summary}</p>
          </li>
        ))}
      </ul>
    </section>
  )
}
