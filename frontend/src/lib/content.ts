import fs from 'node:fs'
import path from 'node:path'

export type PostMetadata = {
  title: string
  publishedAt: string
  summary: string
  repoUrl?: string
  liveUrl?: string
}

export type Post = {
  slug: string
  metadata: PostMetadata
  content: string
}

const FRONTMATTER = /---\s*([\s\S]*?)\s*---/

// Frontmatter is one `key: value` per line (docs/rules.md). Values stay the
// strings they were written as — publishedAt is never parsed into a Date, so
// <time dateTime> and the newest-first sort read it byte-exact.
export function parseFrontmatter(raw: string): {
  metadata: PostMetadata
  content: string
} {
  const match = FRONTMATTER.exec(raw)
  if (!match) {
    throw new Error('missing frontmatter delimiters (---)')
  }

  const metadata: Record<string, string> = {}
  for (const line of match[1].trim().split('\n')) {
    const separator = line.indexOf(': ')
    if (separator === -1) {
      throw new Error(`frontmatter line is not "key: value": ${line}`)
    }

    metadata[line.slice(0, separator).trim()] = line
      .slice(separator + 2)
      .trim()
      .replace(/^['"](.*)['"]$/, '$1')
  }

  return {
    // Required keys and their formats are gated before every build by
    // scripts/validate-content.mjs, which reports the offending file.
    metadata: metadata as PostMetadata,
    content: raw.replace(FRONTMATTER, '').trim(),
  }
}

// The one MDX loader: writing and projects differ only by directory.
// Read at build time only — both collections are fully enumerated by
// generateStaticParams with dynamicParams disabled.
function loadPosts(collection: string): Post[] {
  const directory = path.join(process.cwd(), 'src', 'app', collection, 'posts')

  return fs
    .readdirSync(directory)
    .filter((file) => path.extname(file) === '.mdx')
    .map((file) => ({
      slug: path.basename(file, '.mdx'),
      ...parseFrontmatter(fs.readFileSync(path.join(directory, file), 'utf-8')),
    }))
    .sort((a, b) =>
      b.metadata.publishedAt.localeCompare(a.metadata.publishedAt),
    )
}

export function getWritingPosts(): Post[] {
  return loadPosts('writing')
}

export function getProjects(): Post[] {
  return loadPosts('projects')
}
