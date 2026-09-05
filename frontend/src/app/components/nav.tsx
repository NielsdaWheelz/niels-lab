import Link from 'next/link'
import { ThemeToggle } from './ThemeToggle'

const navItems = [
  ['/writing', 'writing'],
  ['/projects', 'projects'],
  ['/lab', 'lab'],
  ['/log', 'log'],
  ['/now', 'now'],
  ['/cv', 'cv'],
] as const

// The running head (spec §5.1): site title at the canon's text-column
// edge, sections at the same edge's right margin, a hairline beneath. It
// does not mark which section is current — the App Router gives a shared
// root layout no zero-JS way to know the active route without middleware
// (a framework change the spec rules out), and a client component here
// would add a third one beyond the two the colophon confesses to.
export function Navbar() {
  return (
    <nav className="canon site-nav" aria-label="Primary">
      <div className="running-head">
        <Link href="/" className="running-head-name chrome">
          Niels-Erik Nandal
        </Link>
        <div className="running-head-links">
          {navItems.map(([path, name]) => (
            <Link key={path} href={path} className="running-head-link chrome">
              {name}
            </Link>
          ))}
          <ThemeToggle />
        </div>
      </div>
    </nav>
  )
}
