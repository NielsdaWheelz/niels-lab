import { getBlogPosts } from '@/app/blog/utils'
import { getProjects } from '@/app/projects/utils'
import { getBraindumps } from '@/app/braindumps/utils'

export const baseUrl = 'https://niels.dev'

export default async function sitemap() {
  const blogs = getBlogPosts().map((post) => ({
    url: `${baseUrl}/blog/${post.slug}`,
    lastModified: post.metadata.publishedAt,
  }))

  const projects = getProjects().map((project) => ({
    url: `${baseUrl}/projects/${project.slug}`,
    lastModified: project.metadata.publishedAt,
  }))

  const braindumps = getBraindumps().map((dump) => ({
    url: `${baseUrl}/braindumps/${dump.slug}`,
    lastModified: dump.metadata.publishedAt,
  }))

  const routes = ['', '/blog', '/projects', '/braindumps'].map((route) => ({
    url: `${baseUrl}${route}`,
    lastModified: new Date().toISOString().split('T')[0],
  }))

  return [...routes, ...blogs, ...projects, ...braindumps]
}
