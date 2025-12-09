import Link from 'next/link'
import { getProjects, formatDate } from '@/app/projects/utils'

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
    <ul className="projects-list">
      {projects.map((project) => (
        <li key={project.slug} className="project-item">
          <div className="project-content">
            <div className="project-image-card" />
            <div className="project-text">
              <Link href={`/projects/${project.slug}`} className="project-title">
                {project.metadata.title}
              </Link>
              {project.metadata.summary && (
                <p className="project-summary">{project.metadata.summary}</p>
              )}
              <div className="project-meta">
                <time dateTime={project.metadata.publishedAt} className="project-date">
                  {formatDate(project.metadata.publishedAt)}
                </time>
                {project.metadata.repoUrl && (
                  <span className="project-link-sep"> • </span>
                )}
                {project.metadata.repoUrl && (
                  <a 
                    href={project.metadata.repoUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="project-link"
                  >
                    GitHub
                  </a>
                )}
                {project.metadata.liveUrl && (
                  <span className="project-link-sep"> • </span>
                )}
                {project.metadata.liveUrl && (
                  <a 
                    href={project.metadata.liveUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="project-link"
                  >
                    Live
                  </a>
                )}
              </div>
            </div>
          </div>
        </li>
      ))}
    </ul>
  )
}
