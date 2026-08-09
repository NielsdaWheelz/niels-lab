import { notFound } from 'next/navigation'
import { CustomMDX } from '@/app/components/mdx'
import { getWritingPosts } from '@/lib/content'
import { PageTitle } from '@/app/components/PageTitle'
import { JsonLd } from '@/app/components/StructuredData'
import {
  createPageMetadata,
  getCanonicalUrl,
  getPersonSchemaId,
  getWebsiteSchemaId,
} from '@/app/site'
import styles from '@/app/posts.module.css'

// The essays are a closed set: no slug outside generateStaticParams exists,
// so the loader never reads the filesystem at request time.
export const dynamicParams = false

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
      <article>
        <PageTitle>{post.metadata.title}</PageTitle>
        <p className={styles.gloss}>{post.metadata.summary}</p>
        <p className={styles.dateline}>
          <time className="date" dateTime={post.metadata.publishedAt}>
            {post.metadata.publishedAt}
          </time>
        </p>
        <div className={`prose ${styles.body}`}>
          <CustomMDX source={post.content} />
        </div>
      </article>
    </>
  )
}
