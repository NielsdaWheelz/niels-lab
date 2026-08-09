import { getProjects, getWritingPosts } from '@/lib/content'
import { lastWritten, lists } from '@/content/lists'
import { bioExtended, bookTitle, getSiteUrl, siteDescription } from '@/app/site'

export const dynamic = 'force-static'

export async function GET() {
  const siteUrl = getSiteUrl()

  // The loader already sorts newest first (src/lib/content.ts).
  const writingPosts = getWritingPosts()
  const projects = getProjects()

  // Generated from the typed corpus, so the map cannot drift from the pages.
  const listsSection = lists
    .map((list) => {
      const gloss = list.note ? ` ${list.note}` : ''

      return `- [${list.title}](${siteUrl}/lists/${list.slug}): ${list.entries.length} entries, last written ${lastWritten(list)}.${gloss}`
    })
    .join('\n')

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
    `- [Log](${siteUrl}/log): a dated ledger of training, reading, and shipping, newest first; the failures stay in.`,
    `- [CV](${siteUrl}/cv): the record — roles, projects, education, tools; prints on one page.`,
    `- [Now](${siteUrl}/now): a dated diary page — what the work, the reading, and the training are at the moment.`,
    `- [Colophon](${siteUrl}/colophon): how this site is designed and built.`,
    `- [Lab](${siteUrl}/lab): interactive experiments on model internals.`,
    `- [How sampling works](${siteUrl}/lab/sampling): temperature, top-k, and top-p on a live token distribution.`,
    `- [RSS feed](${siteUrl}/rss): the lists, the log, and new writing.`,
    `- [llms-full.txt](${siteUrl}/llms-full.txt): the entire corpus (bio, every list entry, every log row, every project, every writing post, and the CV) as one plain-text document, meant to be read in full by a language model.`,
  ].join('\n')

  const body = `# ${bookTitle}

> ${siteDescription}

${bioExtended}

This site is a book of lists after Sei Shōnagon. Each list is dated, each entry
is one line, and a checkable claim carries a link to its primary source; where
no proof exists the claim is written as plain observation. Read
[llms-full.txt](${siteUrl}/llms-full.txt) for the whole corpus in one document.

## Lists

${listsSection}

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
