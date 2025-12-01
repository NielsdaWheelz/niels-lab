import { notFound } from 'next/navigation'
import { CustomMDX } from '@/app/components/mdx'
import { formatDate, getBraindumps } from '@/app/braindumps/utils'
import { baseUrl } from '@/app/sitemap'

export async function generateStaticParams() {
  const dumps = getBraindumps()

  return dumps.map((dump) => ({
    slug: dump.slug,
  }))
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params
  const dump = getBraindumps().find((d) => d.slug === slug)
  if (!dump) {
    return
  }

  const {
    title,
    publishedAt: publishedTime,
    summary: description,
    image,
  } = dump.metadata
  const ogImage = image
    ? image
    : `${baseUrl}/og?title=${encodeURIComponent(title)}`

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      type: 'article',
      publishedTime,
      url: `${baseUrl}/braindumps/${dump.slug}`,
      images: [
        {
          url: ogImage,
        },
      ],
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      images: [ogImage],
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
      <script
        type="application/ld+json"
        suppressHydrationWarning
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'Article',
            headline: dump.metadata.title,
            datePublished: dump.metadata.publishedAt,
            dateModified: dump.metadata.publishedAt,
            description: dump.metadata.summary,
            image: dump.metadata.image
              ? `${baseUrl}${dump.metadata.image}`
              : `/og?title=${encodeURIComponent(dump.metadata.title)}`,
            url: `${baseUrl}/braindumps/${dump.slug}`,
            author: {
              '@type': 'Person',
              name: 'Niels-Erik Nandal',
            },
          }),
        }}
      />
      <h1 className="title font-semibold text-2xl tracking-tighter">
        {dump.metadata.title}
      </h1>
      <div className="flex justify-between items-center mt-2 mb-8 text-sm">
        <p className="text-sm text-neutral-600 dark:text-neutral-400">
          {formatDate(dump.metadata.publishedAt)}
        </p>
      </div>
      <article className="prose">
        <CustomMDX source={dump.content} />
      </article>
    </section>
  )
}

