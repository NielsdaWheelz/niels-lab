import { BlogPosts } from '@/app/components/posts'
import { PageTitle } from '@/app/components/PageTitle'
import { HomeContent } from '@/app/components/HomeContent'
import { ProjectsList } from '@/app/components/projects'

export default function Page() {
  return (
    <section>
      <PageTitle>niels</PageTitle>
      <HomeContent />
      
      <h2 style={{ marginTop: '2.5rem' }}>projects</h2>
      <ProjectsList limit={3} />
      
      <h2 style={{ marginTop: '2rem' }}>recent writing</h2>
      <BlogPosts />
    </section>
  )
}
