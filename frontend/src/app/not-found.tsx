import Link from 'next/link'
import { PageTitle } from '@/app/components/PageTitle'

export default function NotFound() {
  return (
    <section>
      <PageTitle>404</PageTitle>
      <p className="page-intro">page not found.</p>
      <p>
        <Link href="/">home</Link>
        {' · '}
        <Link href="/projects">projects</Link>
        {' · '}
        <Link href="/writing">writing</Link>
      </p>
    </section>
  )
}
