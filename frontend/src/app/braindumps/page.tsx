import { BraindumpsList } from '@/app/components/braindumps'

export const metadata = {
  title: 'Braindumps',
  description: 'Daily logs and unstructured thoughts.',
}

export default function Page() {
  return (
    <section>
      <h1 className="font-semibold text-2xl mb-8 tracking-tighter">Braindumps</h1>
      <BraindumpsList />
    </section>
  )
}

