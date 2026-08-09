import Link from 'next/link'

export default function Footer() {
  return (
    <footer className="site-footer">
      <p>
        For the record, <Link href="/cv">the CV</Link>. For correspondence,{' '}
        <a href="mailto:niels.erik.nandal@gmail.com">
          niels.erik.nandal@gmail.com
        </a>
        .
      </p>
    </footer>
  )
}
