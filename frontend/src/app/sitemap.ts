import type { MetadataRoute } from 'next'
import { getProjects, getWritingPosts } from '@/lib/content'
import { lastWritten, lists } from '@/content/lists'
import { log } from '@/content/log'
import { getSiteUrl } from '@/app/site'

export default function sitemap(): MetadataRoute.Sitemap {
  const siteUrl = getSiteUrl()

  // The two living pages carry a recrawl signal: the home page moves with
  // the corpus, the log with its newest hand-written row.
  const newestListDate = lists
    .map(lastWritten)
    .reduce((latest, date) => (date > latest ? date : latest))
  const newestLogDate = log
    .map((event) => event.date)
    .reduce((latest, date) => (date > latest ? date : latest))

  const writing = getWritingPosts().map((post) => ({
    url: `${siteUrl}/writing/${post.slug}`,
  }))

  const projects = getProjects().map((project) => ({
    url: `${siteUrl}/projects/${project.slug}`,
  }))

  const listPages = lists.map((list) => ({
    url: `${siteUrl}/lists/${list.slug}`,
    lastModified: lastWritten(list),
  }))

  const routes = [
    { url: `${siteUrl}`, lastModified: newestListDate },
    { url: `${siteUrl}/log`, lastModified: newestLogDate },
    ...[
      '/projects',
      '/writing',
      '/cv',
      '/now',
      '/colophon',
      '/lab',
      '/lab/sampling',
    ].map((route) => ({ url: `${siteUrl}${route}` })),
  ]

  return [...routes, ...listPages, ...projects, ...writing]
}
