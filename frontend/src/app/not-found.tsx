import Link from 'next/link'

export default function NotFound() {
  return (
    <section>
      <h1>404</h1>
      <p>
        This page was torn out of the notebook. Whatever was written here went
        with it — page number and all.
      </p>
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
