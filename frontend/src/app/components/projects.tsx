import Link from 'next/link'
import { getProjects } from '@/app/projects/utils'

export function ProjectsList() {
  const allProjects = getProjects()

  return (
    <ul style={{ listStyle: 'none', padding: 0 }}>
      {allProjects
        .sort((a, b) => {
          if (
            new Date(a.metadata.publishedAt) > new Date(b.metadata.publishedAt)
          ) {
            return -1
          }
          return 1
        })
        .map((project) => (
          <li key={project.slug} style={{ marginBottom: '0.5rem' }}>
            <Link href={`/projects/${project.slug}`}>
              {project.metadata.title}
            </Link>
            {project.metadata.summary && (
              <span style={{ color: '#666' }}> — {project.metadata.summary}</span>
            )}
          </li>
        ))}
    </ul>
  )
}
