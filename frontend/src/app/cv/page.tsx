import { CustomMDX } from '@/app/components/mdx'
import { getCVContent } from '@/app/cv/utils'
import { PageTitle } from '@/app/components/PageTitle'
import { ContentReveal } from '@/app/components/ContentReveal'

export const metadata = {
  title: 'cv',
  description: 'professional experience and background',
}

export default function CVPage() {
  const { content } = getCVContent()

  return (
    <section>
      <PageTitle>cv</PageTitle>
      <ContentReveal loadingText="loading cv">
        <article className="prose">
          <CustomMDX source={content} />
        </article>
      </ContentReveal>
    </section>
  )
}
