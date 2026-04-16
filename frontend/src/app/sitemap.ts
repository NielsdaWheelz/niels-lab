import { getProjects } from '@/app/projects/utils'
import { getWritingPosts } from '@/app/writing/utils'
import { baseUrl } from '@/app/site'

export default function sitemap() {
  const writing = getWritingPosts().map((post) => ({
    url: `${baseUrl}/writing/${post.slug}`,
    lastModified: post.metadata.publishedAt,
  }))

  const projects = getProjects().map((project) => ({
    url: `${baseUrl}/projects/${project.slug}`,
    lastModified: project.metadata.publishedAt,
  }))

  const routes = ['', '/projects', '/writing', '/cv'].map((route) => ({
    url: `${baseUrl}${route}`,
    lastModified: new Date().toISOString().split('T')[0],
  }))

  return [...routes, ...projects, ...writing]
}
