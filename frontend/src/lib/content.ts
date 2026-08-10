import fs from 'node:fs'
import path from 'node:path'

export type PostMetadata = {
  title: string
  publishedAt: string
  summary: string
  repoUrl?: string
  liveUrl?: string
  // 'true' (the only value the validator admits) marks a draft. Drafts never
  // leave collectPosts, so publishedAt and summary are required only of the
  // posts a consumer can see; a draft needs a title and nothing else.
  draft?: string
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

// Pure core of the loader: parse, drop drafts, sort newest first. This is
// the one choke point for post drafts — filtered here, a draft is invisible
// to every consumer: listings, generateStaticParams (its slug 404s),
// sitemap, RSS, llms.txt, llms-full.txt. Drafts go before the sort reads
// publishedAt, which drafts may lack.
export function collectPosts(files: { slug: string; raw: string }[]): Post[] {
  return files
    .map(({ slug, raw }) => ({ slug, ...parseFrontmatter(raw) }))
    .filter((post) => post.metadata.draft !== 'true')
    .sort((a, b) =>
      b.metadata.publishedAt.localeCompare(a.metadata.publishedAt),
    )
}

// The one MDX loader: writing and projects differ only by directory.
// Read at build time only — both collections are fully enumerated by
// generateStaticParams with dynamicParams disabled.
function loadPosts(collection: string): Post[] {
  const directory = path.join(process.cwd(), 'src', 'app', collection, 'posts')

  return collectPosts(
    fs
      .readdirSync(directory)
      .filter((file) => path.extname(file) === '.mdx')
      .map((file) => ({
        slug: path.basename(file, '.mdx'),
        raw: fs.readFileSync(path.join(directory, file), 'utf-8'),
      })),
  )
}

export function getWritingPosts(): Post[] {
  return loadPosts('writing')
}

export function getProjects(): Post[] {
  return loadPosts('projects')
}
