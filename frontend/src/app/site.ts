import type { Metadata } from 'next'

const DEFAULT_SITE_URL = 'https://nielseriknandal.com'

function readEnv(name: string) {
  if (typeof process === 'undefined') {
    return undefined
  }

  return process.env[name]
}

function normalizeSiteUrl(value: string | undefined, assumeHttps = false) {
  if (!value) {
    return null
  }

  const trimmed = value.trim()
  if (!trimmed) {
    return null
  }

  const candidate =
    trimmed.startsWith('http://') || trimmed.startsWith('https://')
      ? trimmed
      : assumeHttps
        ? `https://${trimmed}`
        : trimmed

  try {
    return new URL(candidate).origin
  } catch {
    return null
  }
}

export const siteName = 'Niels Erik Nandal'
export const siteDescription =
  'AI systems engineer building ambitious products with deterministic backends, legible interfaces, and readable software.'
export const githubUrl = 'https://github.com/NielsdaWheelz'
export const linkedinUrl = 'https://www.linkedin.com/in/nielseriknandal/'
export const xUrl = 'https://x.com/the_powertool'
export const huggingFaceUrl = 'https://huggingface.co/nnandal'
export const xHandle = '@the_powertool'
export const socialProfileUrls = [githubUrl, linkedinUrl, xUrl, huggingFaceUrl]

export function getSiteUrl() {
  return (
    normalizeSiteUrl(readEnv('NEXT_PUBLIC_SITE_URL')) ??
    normalizeSiteUrl(readEnv('VERCEL_PROJECT_PRODUCTION_URL'), true) ??
    (readEnv('VERCEL_ENV') === 'production'
      ? normalizeSiteUrl(readEnv('VERCEL_URL'), true)
      : null) ??
    DEFAULT_SITE_URL
  )
}

export const baseUrl = getSiteUrl()

export function getCanonicalUrl(pathname = '/') {
  return new URL(pathname, `${getSiteUrl()}/`).toString()
}

export function getPersonSchemaId() {
  return `${getCanonicalUrl('/')}#person`
}

export function getWebsiteSchemaId() {
  return `${getCanonicalUrl('/')}#website`
}

export function shouldIndexSite() {
  const vercelEnv = readEnv('VERCEL_ENV')

  if (vercelEnv) {
    return vercelEnv === 'production'
  }

  return getSiteUrl() === DEFAULT_SITE_URL
}

export function getOgImageUrl(title?: string, description?: string) {
  const url = new URL('/og', getSiteUrl())

  if (title) {
    url.searchParams.set('title', title)
  }
  if (description) {
    url.searchParams.set('description', description)
  }

  return url.toString()
}

type PageMetadataOptions = {
  title: string
  description: string
  path: string
  type?: 'website' | 'article'
  publishedTime?: string
}

export function createPageMetadata({
  title,
  description,
  path,
  type = 'website',
  publishedTime,
}: PageMetadataOptions): Metadata {
  const canonicalUrl = getCanonicalUrl(path)
  const image = getOgImageUrl(title, description)

  return {
    title,
    description,
    authors: [{ name: siteName, url: getCanonicalUrl('/') }],
    creator: siteName,
    publisher: siteName,
    alternates: {
      canonical: canonicalUrl,
      types: {
        'application/rss+xml': getCanonicalUrl('/rss'),
      },
    },
    openGraph: {
      title,
      description,
      url: canonicalUrl,
      siteName,
      locale: 'en_US',
      type,
      ...(type === 'article' && publishedTime
        ? {
            publishedTime,
            authors: [getCanonicalUrl('/')],
          }
        : {}),
      images: [
        {
          url: image,
          width: 1200,
          height: 630,
          alt: `${title} — ${siteName}`,
        },
      ],
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      creator: xHandle,
      images: [image],
    },
  }
}
