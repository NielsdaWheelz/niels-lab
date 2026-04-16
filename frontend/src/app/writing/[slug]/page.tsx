import { notFound } from 'next/navigation'
import { CustomMDX } from '@/app/components/mdx'
import { formatDate, getWritingPosts } from '@/app/writing/utils'
import { PageTitle } from '@/app/components/PageTitle'
import { ContentReveal } from '@/app/components/ContentReveal'
import { baseUrl, getOgImageUrl } from '@/app/site'

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
  const image = getOgImageUrl(title)

  return {
    title,
    description,
    alternates: {
      canonical: `/writing/${post.slug}`,
    },
    openGraph: {
      title,
      description,
      type: 'article',
      publishedTime,
      url: `${baseUrl}/writing/${post.slug}`,
      images: [
        {
          url: image,
          width: 1200,
          height: 630,
          alt: title,
        },
      ],
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      images: [image],
    },
  }
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

  return (
    <section>
      <PageTitle>{post.metadata.title}</PageTitle>
      <p className="article-meta">{formatDate(post.metadata.publishedAt)}</p>
      <ContentReveal loadingText="loading post">
        <article className="prose article-body">
          <CustomMDX source={post.content} />
        </article>
      </ContentReveal>
    </section>
  )
}
