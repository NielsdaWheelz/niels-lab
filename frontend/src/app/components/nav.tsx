'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { ThemeToggle } from './ThemeToggle'

const navItems = [
  ['/projects', 'projects'],
  ['/writing', 'writing'],
  ['/cv', 'cv'],
] as const

export function Navbar() {
  const pathname = usePathname()

  return (
    <nav className="site-nav" aria-label="Primary navigation">
      <Link
        href="/"
        className="nav-mark"
        aria-label="Niels Erik Nandal, home"
        aria-current={pathname === '/' ? 'page' : undefined}
      >
        <span className="nav-monogram" aria-hidden="true">
          N<span>/</span>E
        </span>
        <span className="nav-wordmark">
          niels
          <small>systems notebook</small>
        </span>
      </Link>
      <div className="nav-links">
        {navItems.map(([path, name]) => {
          const isCurrent = pathname === path || pathname.startsWith(`${path}/`)

          return (
            <Link
              key={path}
              href={path}
              className="nav-link"
              aria-current={isCurrent ? 'page' : undefined}
            >
              {name}
            </Link>
          )
        })}
      </div>
      <ThemeToggle />
    </nav>
  )
}
