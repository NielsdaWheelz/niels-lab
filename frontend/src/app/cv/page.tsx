import Link from 'next/link'
import { CustomMDX } from '@/app/components/mdx'
import { getCVContent } from '@/app/cv/utils'
import { PageTitle } from '@/app/components/PageTitle'
import { ContentReveal } from '@/app/components/ContentReveal'

export const metadata = {
  title: 'cv',
  description: 'professional experience and background',
}

export default function CVPage() {
  const { metadata, content } = getCVContent()

  return (
    <section>
      <PageTitle>cv</PageTitle>
      <p className="page-intro">{metadata.summary}</p>
      <p style={{ marginTop: '0.5rem', marginBottom: '1.5rem', fontSize: '0.9em' }}>
        <Link href="/niels-erik-nandal-cv.pdf" target="_blank" rel="noopener noreferrer">
          view/download pdf
        </Link>
      </p>
      <ContentReveal loadingText="loading cv">
        <article className="prose article-body">
          <CustomMDX source={content} />
        </article>
      </ContentReveal>
    </section>
  )
}
