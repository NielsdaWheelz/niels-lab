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

export function Navbar() {
  return (
    <nav className="site-nav" aria-label="Primary">
      <Link href="/" className="nav-name">
        Niels-Erik Nandal
      </Link>
      {navItems.map(([path, name]) => (
        <Link key={path} href={path} className="nav-link">
          {name}
        </Link>
      ))}
      <ThemeToggle />
    </nav>
  )
}
