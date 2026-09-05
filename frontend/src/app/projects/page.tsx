import Link from 'next/link'
import { getProjects } from '@/lib/content'
import { PageTitle } from '@/app/components/PageTitle'
import { createPageMetadata } from '@/app/site'
import styles from '@/app/posts.module.css'

export const metadata = createPageMetadata({
  title: 'Projects',
  description:
    'Working notes per project: what was built, what held, what broke. By Niels-Erik Nandal.',
  path: '/projects',
})

export default function Page() {
  return (
    <section className="canon leaf">
      <PageTitle>projects</PageTitle>
      <ul className={styles.index} role="list">
        {getProjects().map((project) => (
          <li key={project.slug} className={styles.row}>
            <div className={styles.head}>
              <h2 className="list-title">
                <Link href={`/projects/${project.slug}`}>
                  {project.metadata.title}
                </Link>
              </h2>
              <time className="date" dateTime={project.metadata.publishedAt}>
                {project.metadata.publishedAt}
              </time>
            </div>
            <p className={styles.summary}>{project.metadata.summary}</p>
          </li>
        ))}
      </ul>
    </section>
  )
}
