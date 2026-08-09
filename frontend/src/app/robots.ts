import type { MetadataRoute } from 'next'
import { getSiteUrl, shouldIndexSite } from '@/app/site'

export default function robots(): MetadataRoute.Robots {
  const siteUrl = getSiteUrl()
  const shouldIndex = shouldIndexSite()

  return {
    rules: shouldIndex
      ? {
          userAgent: '*',
          allow: '/',
        }
      : {
          userAgent: '*',
          disallow: '/',
        },
    sitemap: `${siteUrl}/sitemap.xml`,
  }
}
