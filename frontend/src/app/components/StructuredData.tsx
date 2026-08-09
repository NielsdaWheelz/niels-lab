import {
  bio,
  getCanonicalUrl,
  getPersonSchemaId,
  getWebsiteSchemaId,
  siteDescription,
  siteName,
  socialProfileUrls,
} from '@/app/site'

type JsonLdProps = {
  data: Record<string, unknown>
}

export function JsonLd({ data }: JsonLdProps) {
  const json = JSON.stringify(data)

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: json.replace(/</g, '\\u003c') }}
    />
  )
}

export function StructuredData() {
  const siteUrl = getCanonicalUrl('/')
  const personId = getPersonSchemaId()
  const websiteId = getWebsiteSchemaId()
  const personSchema = {
    '@type': 'Person',
    '@id': personId,
    name: siteName,
    url: siteUrl,
    jobTitle: 'Senior Software Engineer',
    worksFor: {
      '@type': 'Organization',
      name: 'Solid',
      url: 'https://solid.tech',
    },
    // Byte-identical to the visible bio; the string lives once, in site.ts.
    description: bio,
    knowsAbout: [
      'Backend systems',
      'Language models',
      'Python',
      'TypeScript',
      'PostgreSQL',
    ],
    sameAs: socialProfileUrls,
  }

  const websiteSchema = {
    '@type': 'WebSite',
    '@id': websiteId,
    name: siteName,
    url: siteUrl,
    description: siteDescription,
    inLanguage: 'en-US',
    publisher: { '@id': personId },
  }

  return (
    <JsonLd
      data={{
        '@context': 'https://schema.org',
        '@graph': [personSchema, websiteSchema],
      }}
    />
  )
}
