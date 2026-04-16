import { notFound } from 'next/navigation'
import { CustomMDX } from '@/app/components/mdx'
import { formatDate, getWritingPosts } from '@/app/writing/utils'
import { PageTitle } from '@/app/components/PageTitle'
import { ContentReveal } from '@/app/components/ContentReveal'
import { baseUrl } from '@/app/site'

export async function generateStaticParams() {
  const posts = getWritingPosts()
  return posts.map((post) => ({ slug: post.slug }))
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params
  const post = getWritingPosts().find((post) => post.slug === slug)
  if (!post) return

  const { title, publishedAt: publishedTime, summary: description } = post.metadata

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      type: 'article',
      publishedTime,
      url: `${baseUrl}/writing/${post.slug}`,
    },
  }
}

export default async function WritingPost({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params
  const post = getWritingPosts().find((post) => post.slug === slug)

  if (!post) {
    notFound()
  }

  return (
    <section>
      <PageTitle>{post.metadata.title}</PageTitle>
      <p className="article-meta">
        {formatDate(post.metadata.publishedAt)}
      </p>
      <ContentReveal loadingText="loading post">
        <article className="prose article-body">
          <CustomMDX source={post.content} />
        </article>
      </ContentReveal>
    </section>
  )
}
