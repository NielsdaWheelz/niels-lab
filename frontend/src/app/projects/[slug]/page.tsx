import Image from 'next/image'
import { notFound } from 'next/navigation'
import { CustomMDX } from '@/app/components/mdx'
import { formatDate, getProjects } from '@/app/projects/utils'
import { PageTitle } from '@/app/components/PageTitle'
import { JsonLd } from '@/app/components/StructuredData'
import {
  createPageMetadata,
  getCanonicalUrl,
  getPersonSchemaId,
  getWebsiteSchemaId,
} from '@/app/site'

export async function generateStaticParams() {
  const projects = getProjects()
  return projects.map((project) => ({ slug: project.slug }))
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const project = getProjects().find((p) => p.slug === slug)
  if (!project) return

  const { title, summary: description } = project.metadata

  return createPageMetadata({
    title,
    description,
    path: `/projects/${project.slug}`,
    type: 'article',
    publishedTime: project.metadata.publishedAt,
  })
}

export default async function Project({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const project = getProjects().find((p) => p.slug === slug)

  if (!project) {
    notFound()
  }

  const projectUrl = getCanonicalUrl(`/projects/${project.slug}`)
  const projectSchema = {
    '@context': 'https://schema.org',
    '@type': 'SoftwareSourceCode',
    '@id': `${projectUrl}#project`,
    name: project.metadata.title,
    description: project.metadata.summary,
    url: projectUrl,
    image: getCanonicalUrl(project.metadata.image),
    datePublished: project.metadata.publishedAt,
    inLanguage: 'en-US',
    author: { '@id': getPersonSchemaId() },
    creator: { '@id': getPersonSchemaId() },
    isPartOf: { '@id': getWebsiteSchemaId() },
    mainEntityOfPage: projectUrl,
    ...(project.metadata.repoUrl
      ? { codeRepository: project.metadata.repoUrl }
      : {}),
    ...(project.metadata.liveUrl
      ? { relatedLink: project.metadata.liveUrl }
      : {}),
  }

  return (
    <>
      <JsonLd data={projectSchema} />
      <section>
        <PageTitle>{project.metadata.title}</PageTitle>
        <p className="article-summary">{project.metadata.summary}</p>
        <p className="article-meta">
          <time dateTime={project.metadata.publishedAt}>
            {formatDate(project.metadata.publishedAt)}
          </time>
        </p>
        <Image
          src={project.metadata.image}
          alt={`Overview graphic for ${project.metadata.title}`}
          width={1200}
          height={800}
          className="article-image"
        />
        {(project.metadata.repoUrl || project.metadata.liveUrl) && (
          <p className="article-links">
            {project.metadata.repoUrl && (
              <a
                href={project.metadata.repoUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                repo
              </a>
            )}
            {project.metadata.repoUrl && project.metadata.liveUrl && ' · '}
            {project.metadata.liveUrl && (
              <a
                href={project.metadata.liveUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                live
              </a>
            )}
          </p>
        )}
        <article className="prose article-body">
          <CustomMDX source={project.content} />
        </article>
      </section>
    </>
  )
}
