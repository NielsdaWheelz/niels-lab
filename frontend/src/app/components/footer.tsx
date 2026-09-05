import Link from 'next/link'

export default function Footer() {
  return (
    <footer className="canon site-footer">
      <p>
        For the record, <Link href="/cv">the CV</Link>. For correspondence,{' '}
        <a href="mailto:niels.erik.nandal@gmail.com">
          niels.erik.nandal@gmail.com
        </a>
        . For the making of it, <Link href="/colophon">the colophon</Link>.
      </p>
    </footer>
  )
}
