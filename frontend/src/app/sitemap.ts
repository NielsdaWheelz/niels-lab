import type { MetadataRoute } from 'next'
import { getProjects } from '@/app/projects/utils'
import { getWritingPosts } from '@/app/writing/utils'
import { getSiteUrl } from '@/app/site'

export default function sitemap(): MetadataRoute.Sitemap {
  const siteUrl = getSiteUrl()

  const writing = getWritingPosts().map((post) => ({
    url: `${siteUrl}/writing/${post.slug}`,
  }))

  const projects = getProjects().map((project) => ({
    url: `${siteUrl}/projects/${project.slug}`,
  }))

  const routes = ['', '/projects', '/writing', '/cv'].map((route) => ({
    url: `${siteUrl}${route}`,
  }))

  return [...routes, ...projects, ...writing]
}
