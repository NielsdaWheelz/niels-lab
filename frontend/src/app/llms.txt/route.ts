import { getWritingPosts } from '@/app/writing/utils'
import { getProjects } from '@/app/projects/utils'
import { getSiteUrl, siteDescription, siteName } from '@/app/site'

export const dynamic = 'force-static'

export async function GET() {
  const siteUrl = getSiteUrl()

  const writingPosts = [...getWritingPosts()].sort((a, b) =>
    b.metadata.publishedAt.localeCompare(a.metadata.publishedAt),
  )
  const projects = [...getProjects()].sort((a, b) =>
    b.metadata.publishedAt.localeCompare(a.metadata.publishedAt),
  )

  const writingSection = writingPosts
    .map(
      (post) =>
        `- [${post.metadata.title}](${siteUrl}/writing/${post.slug}): ${post.metadata.summary}`,
    )
    .join('\n')

  const projectsSection = projects
    .map(
      (project) =>
        `- [${project.metadata.title}](${siteUrl}/projects/${project.slug}): ${project.metadata.summary}`,
    )
    .join('\n')

  const pagesSection = [
    `- [CV](${siteUrl}/cv): full career history, skills, and experience as a structured resume.`,
    `- [Now](${siteUrl}/now): what Niels is doing these days.`,
    `- [Colophon](${siteUrl}/colophon): how this site is designed and built.`,
    `- [Lab](${siteUrl}/lab): interactive experiments on model internals.`,
    `- [RSS feed](${siteUrl}/rss): subscribe to new writing posts.`,
    `- [llms-full.txt](${siteUrl}/llms-full.txt): the entire corpus (bio, every project, every writing post, and the CV) as one plain-text document, meant to be read in full by a language model.`,
  ].join('\n')

  const body = `# ${siteName}

> ${siteDescription}

Niels Erik Nandal is an AI systems engineer: he designs deterministic backends,
builds legible interfaces on top of them, and writes software meant to be read,
not just run. This site is his portfolio — production projects, technical
writing, and a structured CV — and it is built to be as legible to a language
model as it is to a person. Start here for an index of everything on the site,
or read [llms-full.txt](${siteUrl}/llms-full.txt) for the complete corpus in a
single document.

## Writing

${writingSection}

## Projects

${projectsSection}

## Pages

${pagesSection}
`

  return new Response(body, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
    },
  })
}
