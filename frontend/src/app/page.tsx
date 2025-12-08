import { BlogPosts } from '@/app/components/posts'
import { ProjectsList } from '@/app/components/projects'
import { Hero } from '@/app/components/Hero'

export default function Page() {
  return (
    <section>
      <Hero />
      
      <h2 style={{ marginTop: '1.5rem' }}>projects</h2>
      <ProjectsList limit={3} />
      
      <h2 style={{ marginTop: '2rem' }}>recent writing</h2>
      <BlogPosts />
    </section>
  )
}
