import { notFound } from 'next/navigation'
import { CustomMDX } from '@/app/components/mdx'
import { formatDate, getBraindumps } from '@/app/braindumps/utils'
import { baseUrl } from '@/app/sitemap'
import { PageTitle } from '@/app/components/PageTitle'
import { ContentReveal } from '@/app/components/ContentReveal'

export async function generateStaticParams() {
  const dumps = getBraindumps()
  return dumps.map((dump) => ({ slug: dump.slug }))
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params
  const dump = getBraindumps().find((d) => d.slug === slug)
  if (!dump) return

  const { title, summary: description } = dump.metadata

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      url: `${baseUrl}/braindumps/${dump.slug}`,
    },
  }
}

export default async function Braindump({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params
  const dump = getBraindumps().find((d) => d.slug === slug)

  if (!dump) {
    notFound()
  }

  return (
    <section>
      <PageTitle>{dump.metadata.title}</PageTitle>
      <p style={{ color: '#666', marginBottom: '2rem' }}>
        {formatDate(dump.metadata.publishedAt)}
      </p>
      <ContentReveal loadingText="loading">
        <article className="prose">
          <CustomMDX source={dump.content} />
        </article>
      </ContentReveal>
    </section>
  )
}
