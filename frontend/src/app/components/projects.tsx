import Image from 'next/image'
import Link from 'next/link'
import { getProjects, formatDate } from '@/app/projects/utils'

interface ProjectsListProps {
  slugs?: string[]
  variant?: 'list' | 'showcase'
}

const projectNotes: Record<string, { discipline: string; signal: string }> = {
  'niels-gpt': {
    discipline: 'model systems',
    signal: 'raw text → tokenizer → transformer → chat',
  },
  nexus: {
    discipline: 'knowledge systems',
    signal: 'ingest → anchor → retrieve → reason',
  },
  'factory-simulator': {
    discipline: 'agent systems',
    signal: 'interpret → schedule → validate → trace',
  },
  'suno-demo': {
    discipline: 'generative interfaces',
    signal: 'generate → embed → cluster → branch',
  },
  'fractal-chat': {
    discipline: 'retrieval products',
    signal: 'ingest → chunk → ground → converse',
  },
  'fractal-go': {
    discipline: 'realtime systems',
    signal: 'play → synchronize → validate → persist',
  },
}

export function ProjectsList({ slugs, variant = 'list' }: ProjectsListProps) {
  const allProjects = getProjects().sort((a, b) => {
    if (new Date(a.metadata.publishedAt) > new Date(b.metadata.publishedAt)) {
      return -1
    }
    return 1
  })

  const projects = slugs
    ? slugs.flatMap((slug) => {
        const project = allProjects.find((candidate) => candidate.slug === slug)
        return project ? [project] : []
      })
    : allProjects

  return (
    <ul className={`projects-list projects-list--${variant}`}>
      {projects.map((project, index) => {
        const note = projectNotes[project.slug]

        return (
          <li key={project.slug} className="project-item">
            <div className="project-content">
              <Link
                href={`/projects/${project.slug}`}
                className="project-image-card"
                aria-label={`Read the ${project.metadata.title} case study`}
              >
                <span className="project-card-number" aria-hidden="true">
                  {String(index + 1).padStart(2, '0')}
                </span>
                <Image
                  src={project.metadata.image}
                  alt=""
                  width={1200}
                  height={800}
                  preload={variant === 'showcase' && index === 0}
                  className="project-image"
                />
              </Link>
              <div className="project-text">
                {note && (
                  <p className="project-discipline">{note.discipline}</p>
                )}
                <Link
                  href={`/projects/${project.slug}`}
                  className="project-title"
                >
                  {project.metadata.title}
                </Link>
                {project.metadata.summary && (
                  <p className="project-summary">{project.metadata.summary}</p>
                )}
                {note && <p className="project-signal">{note.signal}</p>}
                <div className="project-meta">
                  <time
                    dateTime={project.metadata.publishedAt}
                    className="project-date"
                  >
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
                      source ↗
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
                      live ↗
                    </a>
                  )}
                </div>
              </div>
            </div>
          </li>
        )
      })}
    </ul>
  )
}
