import { BlogPosts } from '@/app/components/posts'
import { PageTitle } from '@/app/components/PageTitle'
import { HomeContent } from '@/app/components/HomeContent'

export default function Page() {
  return (
    <section>
      <PageTitle>niels</PageTitle>
      <HomeContent />
      <h2 style={{ marginTop: '2rem' }}>recent</h2>
      <BlogPosts />
    </section>
  )
}
