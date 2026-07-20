import {
  siteName,
  siteDescription,
  githubUrl,
  linkedinUrl,
  xUrl,
  getSiteUrl,
} from '@/app/site'

export function StructuredData() {
  const siteUrl = getSiteUrl()

  const personSchema = {
    '@context': 'https://schema.org',
    '@type': 'Person',
    name: siteName,
    url: siteUrl,
    jobTitle: 'Software Engineer',
    description: siteDescription,
    sameAs: [githubUrl, linkedinUrl, xUrl],
  }

  const websiteSchema = {
    '@context': 'https://schema.org',
    '@type': 'WebSite',
    name: siteName,
    url: siteUrl,
    description: siteDescription,
  }

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(personSchema) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(websiteSchema) }}
      />
    </>
  )
}
