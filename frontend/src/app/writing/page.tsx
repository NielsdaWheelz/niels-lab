import Link from 'next/link'
import { getWritingPosts } from '@/lib/content'
import { PageTitle } from '@/app/components/PageTitle'
import { createPageMetadata } from '@/app/site'
import styles from '@/app/posts.module.css'

export const metadata = createPageMetadata({
  title: 'Writing',
  description:
    'Essays: neural networks explained from zero, in plain English. By Niels-Erik Nandal.',
  path: '/writing',
})

export default function Page() {
  return (
    <section className="canon leaf">
      <PageTitle>writing</PageTitle>
      <ul className={styles.index} role="list">
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
