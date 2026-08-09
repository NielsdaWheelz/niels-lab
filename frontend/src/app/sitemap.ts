import type { MetadataRoute } from 'next'
import { getProjects, getWritingPosts } from '@/lib/content'
import { lastWritten, lists } from '@/content/lists'
import { getSiteUrl } from '@/app/site'

export default function sitemap(): MetadataRoute.Sitemap {
  const siteUrl = getSiteUrl()

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
    '',
    '/log',
    '/projects',
    '/writing',
    '/cv',
    '/now',
    '/colophon',
    '/lab',
    '/lab/sampling',
  ].map((route) => ({
    url: `${siteUrl}${route}`,
  }))

  return [...routes, ...listPages, ...projects, ...writing]
}
