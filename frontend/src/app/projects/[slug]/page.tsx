import { notFound } from 'next/navigation'
import { CustomMDX } from '@/app/components/mdx'
import { getProjects } from '@/lib/content'
import { PageTitle } from '@/app/components/PageTitle'
import { JsonLd } from '@/app/components/StructuredData'
import {
  createPageMetadata,
  getCanonicalUrl,
  getPersonSchemaId,
  getWebsiteSchemaId,
} from '@/app/site'
import styles from '@/app/posts.module.css'

// The projects are a closed set: no slug outside generateStaticParams exists,
// so the loader never reads the filesystem at request time.
export const dynamicParams = false

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
      <article className="canon leaf">
        <PageTitle>{project.metadata.title}</PageTitle>
        <p className={styles.gloss}>{project.metadata.summary}</p>
        <p className={styles.dateline}>
          <time className="date" dateTime={project.metadata.publishedAt}>
            {project.metadata.publishedAt}
          </time>
        </p>
        {project.metadata.repoUrl || project.metadata.liveUrl ? (
          <p className={`chrome ${styles.links}`}>
            {project.metadata.repoUrl ? (
              <a
                href={project.metadata.repoUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                repo
              </a>
            ) : null}
            {project.metadata.repoUrl && project.metadata.liveUrl ? ' · ' : ''}
            {project.metadata.liveUrl ? (
              <a
                href={project.metadata.liveUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                live
              </a>
            ) : null}
          </p>
        ) : null}
        <div className={`prose ${styles.body}`}>
          <CustomMDX source={project.content} />
        </div>
      </article>
    </>
  )
}
