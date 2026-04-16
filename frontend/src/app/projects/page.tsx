import { ProjectsList } from '@/app/components/projects'
import { PageTitle } from '@/app/components/PageTitle'

export const metadata = {
  title: 'projects',
  description: 'Selected engineering work',
}

export default function Page() {
  return (
    <section>
      <PageTitle>projects</PageTitle>
      <p className="page-intro">
        The work here is the clearest picture of how I build: product-forward,
        technically sharp, and explicit about tradeoffs.
      </p>
      <ProjectsList />
    </section>
  )
}
