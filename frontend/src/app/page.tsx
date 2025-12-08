import { BlogPosts } from '@/app/components/posts'
import { ProjectsList } from '@/app/components/projects'
import { Hero } from '@/app/components/Hero'
import { NeuralHeadingSection } from '@/app/components/NeuralHeadingSection'

export default function Page() {
  return (
    <section>
      <Hero />
      
      <NeuralHeadingSection>
        <h2 style={{ marginTop: '1.5rem' }}>projects</h2>
      </NeuralHeadingSection>
      <ProjectsList limit={3} />
      
      <NeuralHeadingSection>
        <h2 style={{ marginTop: '2rem' }}>recent writing</h2>
      </NeuralHeadingSection>
      <BlogPosts />
    </section>
  )
}
