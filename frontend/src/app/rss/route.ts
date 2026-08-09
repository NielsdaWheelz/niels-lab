import { getWritingPosts } from '@/lib/content'
import { lastWritten, lists } from '@/content/lists'
import { getLedger } from '@/lib/ledger'
import {
  bookTitle,
  getCanonicalUrl,
  siteDescription,
  siteName,
} from '@/app/site'

export const dynamic = 'force-static'

// The escape regex can only produce these five characters, so the lookup
// never misses.
const entities: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&apos;',
}

function escapeXml(value: string) {
  return value.replace(/[&<>"']/g, (character) => entities[character])
}

function rssDate(value: string) {
  return new Date(`${value}T00:00:00Z`).toUTCString()
}

type FeedItem = {
  title: string
  url: string
  // Absent: the url is the permalink guid. Present: a stable non-permalink
  // guid, so readers that dedupe on guid still see every row.
  guid?: string
  description: string
  date: string
}

export async function GET() {
  const posts: FeedItem[] = getWritingPosts().map((post) => ({
    title: post.metadata.title,
    url: getCanonicalUrl(`/writing/${post.slug}`),
    description: post.metadata.summary,
    date: post.metadata.publishedAt,
  }))

  const listItems: FeedItem[] = lists.map((list) => {
    const last = lastWritten(list)

    return {
      title: list.title,
      url: getCanonicalUrl(`/lists/${list.slug}`),
      description: `${list.entries.length} entries, last written ${last}.${list.note ? ` ${list.note}` : ''}`,
      date: last,
    }
  })

  // One item per ledger row, so the feed carries the log instead of
  // describing the page; guids are stable per (date, text).
  const logUrl = getCanonicalUrl('/log')
  const logItems: FeedItem[] = (await getLedger()).map((event) => ({
    title: `${event.date} · ${event.kind}`,
    url: logUrl,
    guid: `${logUrl}#${event.date} ${event.text}`,
    description: event.failed
      ? `${event.text} — failed: ${event.failed.lesson}`
      : event.text,
    date: event.date,
  }))

  const items = [...posts, ...listItems, ...logItems].sort((a, b) =>
    b.date.localeCompare(a.date),
  )

  const itemsXml = items
    .map(
      (item) => `<item>
        <title>${escapeXml(item.title)}</title>
        <link>${escapeXml(item.url)}</link>
        <guid isPermaLink="${item.guid ? 'false' : 'true'}">${escapeXml(item.guid ?? item.url)}</guid>
        <description>${escapeXml(item.description)}</description>
        <dc:creator>${escapeXml(siteName)}</dc:creator>
        <pubDate>${rssDate(item.date)}</pubDate>
      </item>`,
    )
    .join('\n')

  const siteUrl = getCanonicalUrl('/')
  const feedUrl = getCanonicalUrl('/rss')
  // posts + lists + log all exist by construction, so items is never empty.
  const lastBuildDate = `<lastBuildDate>${rssDate(items[0].date)}</lastBuildDate>`

  const rssFeed = `<?xml version="1.0" encoding="UTF-8" ?>
  <rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/">
    <channel>
      <title>${escapeXml(bookTitle)}</title>
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
