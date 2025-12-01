import { ProjectsList } from '@/app/components/projects'
import { PageTitle } from '@/app/components/PageTitle'

export const metadata = {
  title: 'projects',
  description: 'things i\'ve built',
}

export default function Page() {
  return (
    <section>
      <PageTitle>projects</PageTitle>
      <ProjectsList />
    </section>
  )
}
