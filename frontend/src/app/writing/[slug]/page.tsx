import { notFound } from 'next/navigation'
import { CustomMDX } from '@/app/components/mdx'
import { formatDate, getWritingPosts } from '@/app/writing/utils'
import { PageTitle } from '@/app/components/PageTitle'
import { JsonLd } from '@/app/components/StructuredData'
import {
  createPageMetadata,
  getCanonicalUrl,
  getPersonSchemaId,
  getWebsiteSchemaId,
} from '@/app/site'

export async function generateStaticParams() {
  const posts = getWritingPosts()
  return posts.map((post) => ({ slug: post.slug }))
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const post = getWritingPosts().find((post) => post.slug === slug)
  if (!post) return

  const {
    title,
    publishedAt: publishedTime,
    summary: description,
  } = post.metadata

  return createPageMetadata({
    title,
    description,
    path: `/writing/${post.slug}`,
    type: 'article',
    publishedTime,
  })
}

export default async function WritingPost({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const post = getWritingPosts().find((post) => post.slug === slug)

  if (!post) {
    notFound()
  }

  const postUrl = getCanonicalUrl(`/writing/${post.slug}`)

  return (
    <>
      <JsonLd
        data={{
          '@context': 'https://schema.org',
          '@type': ['BlogPosting', 'TechArticle'],
          '@id': `${postUrl}#article`,
          headline: post.metadata.title,
          description: post.metadata.summary,
          url: postUrl,
          datePublished: post.metadata.publishedAt,
          inLanguage: 'en-US',
          author: { '@id': getPersonSchemaId() },
          publisher: { '@id': getPersonSchemaId() },
          isPartOf: { '@id': getWebsiteSchemaId() },
          mainEntityOfPage: postUrl,
        }}
      />
      <section>
        <PageTitle>{post.metadata.title}</PageTitle>
        <p className="article-summary">{post.metadata.summary}</p>
        <p className="article-meta">
          <time dateTime={post.metadata.publishedAt}>
            {formatDate(post.metadata.publishedAt)}
          </time>
        </p>
        <article className="prose article-body">
          <CustomMDX source={post.content} />
        </article>
      </section>
    </>
  )
}
