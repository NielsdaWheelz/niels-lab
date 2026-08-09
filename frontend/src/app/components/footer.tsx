import Link from 'next/link'

export default function Footer() {
  return (
    <footer className="site-footer">
      <p>
        Niels-Erik Nandal · <Link href="/cv">cv</Link> ·{' '}
        <a href="mailto:niels.erik.nandal@gmail.com">email</a> ·{' '}
        <Link href="/colophon">colophon</Link>
      </p>
    </footer>
  )
}
