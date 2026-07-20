import { getWritingPosts } from '@/app/writing/utils'
import { getCanonicalUrl, siteDescription, siteName } from '@/app/site'

export const dynamic = 'force-static'

function escapeXml(value: string) {
  return value.replace(/[&<>"']/g, (character) => {
    const entities: Record<string, string> = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&apos;',
    }

    return entities[character] ?? character
  })
}

function rssDate(value: string) {
  return new Date(`${value}T00:00:00Z`).toUTCString()
}

export async function GET() {
  const allPosts = [...getWritingPosts()].sort((a, b) =>
    b.metadata.publishedAt.localeCompare(a.metadata.publishedAt),
  )

  const itemsXml = allPosts
    .map((post) => {
      const postUrl = getCanonicalUrl(`/writing/${post.slug}`)

      return `<item>
        <title>${escapeXml(post.metadata.title)}</title>
        <link>${escapeXml(postUrl)}</link>
        <guid isPermaLink="true">${escapeXml(postUrl)}</guid>
        <description>${escapeXml(post.metadata.summary)}</description>
        <dc:creator>${escapeXml(siteName)}</dc:creator>
        <pubDate>${rssDate(post.metadata.publishedAt)}</pubDate>
      </item>`
    })
    .join('\n')

  const siteUrl = getCanonicalUrl('/')
  const feedUrl = getCanonicalUrl('/rss')
  const lastBuildDate = allPosts[0]
    ? `<lastBuildDate>${rssDate(allPosts[0].metadata.publishedAt)}</lastBuildDate>`
    : ''

  const rssFeed = `<?xml version="1.0" encoding="UTF-8" ?>
  <rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/">
    <channel>
      <title>${escapeXml(`${siteName} — Writing`)}</title>
      <link>${escapeXml(siteUrl)}</link>
      <atom:link href="${escapeXml(feedUrl)}" rel="self" type="application/rss+xml" />
      <description>${escapeXml(siteDescription)}</description>
      <language>en-US</language>
      ${lastBuildDate}
      ${itemsXml}
    </channel>
  </rss>`

  return new Response(rssFeed, {
    headers: {
      'Content-Type': 'application/rss+xml; charset=utf-8',
      'Cache-Control':
        'public, max-age=0, s-maxage=3600, stale-while-revalidate=86400',
    },
  })
}
