import type { MetadataRoute } from 'next'
import { getProjects } from '@/app/projects/utils'
import { getWritingPosts } from '@/app/writing/utils'
import { getSiteUrl } from '@/app/site'

export default function sitemap(): MetadataRoute.Sitemap {
  const siteUrl = getSiteUrl()

  const writing = getWritingPosts().map((post) => ({
    url: `${siteUrl}/writing/${post.slug}`,
    lastModified: post.metadata.publishedAt,
  }))

  const projects = getProjects().map((project) => ({
    url: `${siteUrl}/projects/${project.slug}`,
    lastModified: project.metadata.publishedAt,
  }))

  const routes = ['', '/projects', '/writing', '/cv'].map((route) => ({
    url: `${siteUrl}${route}`,
    lastModified: new Date().toISOString().split('T')[0],
  }))

  return [...routes, ...projects, ...writing]
}
