import { notFound } from 'next/navigation'
import { ListBody } from '@/app/components/ListBody'
import { lastWritten, lists } from '@/content/lists'
import { createPageMetadata } from '@/app/site'

// The corpus is a closed set: no slug outside generateStaticParams exists,
// so unknown list slugs 404 statically.
export const dynamicParams = false

export function generateStaticParams() {
  return lists.map((list) => ({ slug: list.slug }))
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const list = lists.find((candidate) => candidate.slug === slug)
  if (!list) return

  return createPageMetadata({
    title: list.title,
    description:
      list.note ??
      `${list.entries.length} entries, last written ${lastWritten(list)}. A list from the Pillow Book of Niels-Erik Nandal.`,
    path: `/lists/${list.slug}`,
  })
}

export default async function ListPage({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const list = lists.find((candidate) => candidate.slug === slug)

  if (!list) {
    notFound()
  }

  const last = lastWritten(list)

  return (
    <article>
      <header>
        <h1 className="list-title">{list.title}</h1>
        <p className="chrome">
          <span className="count">{list.entries.length} entries</span>
          {' · '}
          <time className="date" dateTime={last}>
            last written {last}
          </time>
        </p>
      </header>
      <hr />
      <ListBody list={list} />
    </article>
  )
}
