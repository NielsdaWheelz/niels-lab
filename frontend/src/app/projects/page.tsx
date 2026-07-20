import { ProjectsList } from '@/app/components/projects'
import { PageTitle } from '@/app/components/PageTitle'
import { createPageMetadata } from '@/app/site'

export const metadata = createPageMetadata({
  title: 'Projects',
  description:
    'Selected AI, machine learning, and software engineering projects by Niels Erik Nandal.',
  path: '/projects',
})

export default function Page() {
  return (
    <section>
      <PageTitle>projects</PageTitle>
      <p className="page-intro">
        The work here is the clearest picture of how I build: product-forward,
        technically sharp, and explicit about tradeoffs.
      </p>
      <ProjectsList variant="showcase" />
    </section>
  )
}
