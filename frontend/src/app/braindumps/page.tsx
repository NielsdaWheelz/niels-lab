import { BraindumpsList } from '@/app/components/braindumps'
import { PageTitle } from '@/app/components/PageTitle'

export const metadata = {
  title: 'braindumps',
  description: 'daily logs and unstructured thoughts',
}

export default function Page() {
  return (
    <section>
      <PageTitle>braindumps</PageTitle>
      <BraindumpsList />
    </section>
  )
}
