const DEFAULT_SITE_URL = 'https://niels.dev'

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
  'Engineer building AI products, systems, and readable software.'
export const githubUrl = 'https://github.com/NielsdaWheelz'
export const linkedinUrl = 'https://www.linkedin.com/in/nielseriknandal/'
export const xUrl = 'https://x.com/the_powertool'

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

export function shouldIndexSite() {
  const vercelEnv = readEnv('VERCEL_ENV')

  if (vercelEnv) {
    return vercelEnv === 'production'
  }

  return getSiteUrl() === DEFAULT_SITE_URL
}

export function getOgImageUrl(title?: string) {
  const url = new URL('/og', getSiteUrl())

  if (title) {
    url.searchParams.set('title', title)
  }

  return url.toString()
}
