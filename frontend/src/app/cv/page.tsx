import Link from 'next/link'
import { PageTitle } from '@/app/components/PageTitle'
import { ContentReveal } from '@/app/components/ContentReveal'
import Timeline from './Timeline'

export const metadata = {
  title: 'cv',
  description: 'professional experience and background',
}

export default function CVPage() {
  return (
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
      <ContentReveal loadingText="loading cv">
        <Timeline />
      </ContentReveal>
    </section>
  )
}
