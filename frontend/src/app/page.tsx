import { BlogPosts } from '@/app/components/posts'
import { ProjectsList } from '@/app/components/projects'
import { Hero } from '@/app/components/Hero'
import { DrawHeading } from '@/app/components/DrawHeading'

export default function Page() {
  return (
    <section>
      <Hero />
      
      <DrawHeading as="h2" underlineColor="sage" delay={2200}>
        projects
      </DrawHeading>
      <ProjectsList limit={3} />
      
      <DrawHeading as="h2" underlineColor="terracotta" delay={500}>
        recent writing
      </DrawHeading>
      <BlogPosts />
    </section>
  )
}
