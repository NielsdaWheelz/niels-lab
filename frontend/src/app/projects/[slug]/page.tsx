import { notFound } from 'next/navigation'
import { CustomMDX } from '@/app/components/mdx'
import { getProjects } from '@/app/projects/utils'
import { baseUrl } from '@/app/sitemap'
import { PageTitle } from '@/app/components/PageTitle'
import { ContentReveal } from '@/app/components/ContentReveal'

export async function generateStaticParams() {
  const projects = getProjects()
  return projects.map((project) => ({ slug: project.slug }))
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params
  const project = getProjects().find((p) => p.slug === slug)
  if (!project) return

  const { title, summary: description } = project.metadata

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      url: `${baseUrl}/projects/${project.slug}`,
    },
  }
}

export default async function Project({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params
  const project = getProjects().find((p) => p.slug === slug)

  if (!project) {
    notFound()
  }

  return (
    <section>
      <PageTitle>{project.metadata.title}</PageTitle>
      <p style={{ color: '#666', marginBottom: '1rem' }}>
        {project.metadata.summary}
      </p>
      {(project.metadata.repoUrl || project.metadata.liveUrl) && (
        <p style={{ marginBottom: '2rem' }}>
          {project.metadata.repoUrl && (
            <a href={project.metadata.repoUrl} target="_blank" rel="noopener noreferrer">
              repo
            </a>
          )}
          {project.metadata.repoUrl && project.metadata.liveUrl && ' · '}
          {project.metadata.liveUrl && (
            <a href={project.metadata.liveUrl} target="_blank" rel="noopener noreferrer">
              live
            </a>
          )}
        </p>
      )}
      <ContentReveal loadingText="loading project">
        <article className="prose">
          <CustomMDX source={project.content} />
        </article>
      </ContentReveal>
    </section>
  )
}
