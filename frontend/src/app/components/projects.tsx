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
          <li key={project.slug} style={{ marginBottom: '0.75rem' }}>
            <Link 
              href={`/projects/${project.slug}`}
              style={{
                display: 'block',
              }}
            >
              <span>{project.metadata.title}</span>
              {project.metadata.summary && (
                <span style={{ color: '#666', display: 'block', fontSize: '0.9em', marginTop: '0.1rem' }}>
                  {project.metadata.summary}
                </span>
              )}
            </Link>
          </li>
        ))}
    </ul>
  )
}
