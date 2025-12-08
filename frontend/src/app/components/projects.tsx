import Link from 'next/link'
import { getProjects } from '@/app/projects/utils'

interface ProjectsListProps {
  limit?: number
}

export function ProjectsList({ limit }: ProjectsListProps) {
  const allProjects = getProjects()
    .sort((a, b) => {
      if (new Date(a.metadata.publishedAt) > new Date(b.metadata.publishedAt)) {
        return -1
      }
      return 1
    })

  const projects = limit ? allProjects.slice(0, limit) : allProjects

  return (
    <ul style={{ listStyle: 'none', padding: 0 }}>
      {projects.map((project) => (
        <li key={project.slug} style={{ marginBottom: '0.75rem' }}>
          <Link 
            href={`/projects/${project.slug}`}
            style={{
              display: 'block',
            }}
          >
            <span>{project.metadata.title}</span>
            {project.metadata.summary && (
              <span style={{ 
                color: 'var(--color-text-muted)', 
                display: 'block', 
                fontSize: '0.9em', 
                marginTop: '0.1rem' 
              }}>
                {project.metadata.summary}
              </span>
            )}
          </Link>
        </li>
      ))}
    </ul>
  )
}
