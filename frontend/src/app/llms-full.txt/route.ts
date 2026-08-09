import { getProjects, getWritingPosts } from '@/lib/content'
import { lastWritten, lists } from '@/content/lists'
import { getLedger } from '@/lib/ledger'
import { entries, skills } from '@/app/cv/data'
import {
  bioExtended,
  bookTitle,
  getSiteUrl,
  githubUrl,
  huggingFaceUrl,
  linkedinUrl,
  siteDescription,
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
  const ledger = await getLedger()

  const projects = [...getProjects()].sort((a, b) =>
    b.metadata.publishedAt.localeCompare(a.metadata.publishedAt),
  )
  const writingPosts = [...getWritingPosts()].sort((a, b) =>
    b.metadata.publishedAt.localeCompare(a.metadata.publishedAt),
  )

  // Every entry and every proof, from the same typed data the pages render.
  const listsSection = lists
    .map((list) => {
      const entryLines = list.entries
        .map((entry) => {
          if (!entry.evidence) {
            return `- ${entry.text}`
          }

          const proof =
            'href' in entry.evidence ? entry.evidence.href : entry.evidence.src
          // A machine reading this file has no base URL: every proof is absolute.
          const target = proof.startsWith('/') ? `${siteUrl}${proof}` : proof

          return `- ${entry.text}\n  - Evidence: ${entry.evidence.label} — ${target}`
        })
        .join('\n')

      return `### ${list.title}

**URL:** ${siteUrl}/lists/${list.slug}
**Entries:** ${list.entries.length}
**Last written:** ${lastWritten(list)}${list.note ? `\n**Note:** ${list.note}` : ''}

${entryLines}`
    })
    .join('\n\n---\n\n')

  const logSection = ledger
    .map((event) => {
      const lesson = event.failed ? ` — failed: ${event.failed.lesson}` : ''
      const source = event.href ? ` — ${event.href}` : ''

      return `- ${event.date} (${event.kind}) ${event.text}${lesson}${source}`
    })
    .join('\n')

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

  const body = `# ${bookTitle} — Full Corpus

> ${siteDescription}

This document contains the entire readable corpus of ${siteUrl} — bio, every
list entry with its evidence, every log row, every project, every writing post,
and the CV — concatenated as one plain-text file for language models. For a
short index instead, see [llms.txt](${siteUrl}/llms.txt).

## About

${bioExtended}

## Lists

${listsSection}

## Log

${logSection}

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
