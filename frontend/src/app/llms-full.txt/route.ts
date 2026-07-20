import { getWritingPosts } from '@/app/writing/utils'
import { getProjects } from '@/app/projects/utils'
import { entries, skills } from '@/app/cv/data'
import {
  getSiteUrl,
  githubUrl,
  huggingFaceUrl,
  linkedinUrl,
  siteDescription,
  siteName,
  xUrl,
} from '@/app/site'

export const dynamic = 'force-static'

// Strips lines that are pure JSX component tags (e.g. `<Component ... />`)
// left over from MDX source, without touching prose or markdown links.
function stripJsxLines(content: string) {
  return content
    .split('\n')
    .filter((line) => !/^\s*<[A-Z][^>]*\/?>\s*$/.test(line))
    .join('\n')
}

function renderCvEntry(entry: (typeof entries)[number]) {
  const heading =
    'subtitle' in entry ? `${entry.title} — ${entry.subtitle}` : entry.title
  const bullets =
    'bullets' in entry && entry.bullets
      ? entry.bullets.map((bullet) => `- ${bullet}`).join('\n')
      : null

  return [`### ${heading}`, `${entry.date} (${entry.category})`, bullets]
    .filter(Boolean)
    .join('\n\n')
}

function renderCv() {
  const timeline = entries.map(renderCvEntry).join('\n\n')
  const skillLines = Object.entries(skills)
    .map(([category, items]) => `- **${category}:** ${items.join(', ')}`)
    .join('\n')

  return `${timeline}\n\n### Skills\n\n${skillLines}`
}

export async function GET() {
  const siteUrl = getSiteUrl()

  const projects = [...getProjects()].sort((a, b) =>
    b.metadata.publishedAt.localeCompare(a.metadata.publishedAt),
  )
  const writingPosts = [...getWritingPosts()].sort((a, b) =>
    b.metadata.publishedAt.localeCompare(a.metadata.publishedAt),
  )

  const projectsSection = projects
    .map((project) => {
      const url = `${siteUrl}/projects/${project.slug}`

      return `### ${project.metadata.title}

**URL:** ${url}
**Date:** ${project.metadata.publishedAt}
**Summary:** ${project.metadata.summary}

${stripJsxLines(project.content)}`
    })
    .join('\n\n---\n\n')

  const writingSection = writingPosts
    .map((post) => {
      const url = `${siteUrl}/writing/${post.slug}`

      return `### ${post.metadata.title}

**URL:** ${url}
**Date:** ${post.metadata.publishedAt}
**Summary:** ${post.metadata.summary}

${stripJsxLines(post.content)}`
    })
    .join('\n\n---\n\n')

  const body = `# ${siteName} — Full Corpus

> ${siteDescription}

This document contains the entire readable corpus of ${siteUrl} — bio,
every project, every writing post, and the CV — concatenated as one
plain-text file for language models. For a short index instead, see
[llms.txt](${siteUrl}/llms.txt).

## About

Niels Erik Nandal is an AI systems engineer. He builds deterministic backends,
legible interfaces, and readable software, and this site is the record of
that work: production projects with real architecture and evidence behind
them, technical writing about what building them taught him, and a
structured CV covering his education, experience, and skills.

## Projects

${projectsSection}

## Writing

${writingSection}

## CV

${renderCv()}

## Contact

- Email: niels.erik.nandal@gmail.com
- GitHub: ${githubUrl}
- LinkedIn: ${linkedinUrl}
- X: ${xUrl}
- Hugging Face: ${huggingFaceUrl}
`

  return new Response(body, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
    },
  })
}
