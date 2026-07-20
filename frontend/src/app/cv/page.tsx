import './print.css'
import Link from 'next/link'
import { PageTitle } from '@/app/components/PageTitle'
import { JsonLd } from '@/app/components/StructuredData'
import {
  createPageMetadata,
  getCanonicalUrl,
  getPersonSchemaId,
  getWebsiteSchemaId,
  siteName,
} from '@/app/site'
import Timeline from './Timeline'

const description =
  'Software engineering experience, education, projects, and technical skills for Niels Erik Nandal.'

export const metadata = createPageMetadata({
  title: 'CV',
  description,
  path: '/cv',
})

export default function CVPage() {
  const cvUrl = getCanonicalUrl('/cv')

  return (
    <>
      <JsonLd
        data={{
          '@context': 'https://schema.org',
          '@type': 'ProfilePage',
          '@id': `${cvUrl}#profile`,
          url: cvUrl,
          name: `CV — ${siteName}`,
          description,
          inLanguage: 'en-US',
          isPartOf: { '@id': getWebsiteSchemaId() },
          mainEntity: { '@id': getPersonSchemaId() },
          about: { '@id': getPersonSchemaId() },
        }}
      />
      <section>
        <PageTitle>cv</PageTitle>
        <p className="page-intro">Professional experience and background</p>
        <p
          style={{
            marginTop: '0.5rem',
            marginBottom: '1.5rem',
            fontSize: '0.9em',
          }}
        >
          <Link
            href="/niels-erik-nandal-cv.pdf"
            target="_blank"
            rel="noopener noreferrer"
          >
            view/download pdf
          </Link>
        </p>
        <Timeline />
      </section>
    </>
  )
}
