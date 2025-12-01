import Link from 'next/link'

const navItems = {
  '/': 'home',
  '/blog': 'blog',
  '/projects': 'projects',
  '/braindumps': 'braindumps',
}

export function Navbar() {
  return (
    <nav>
      {Object.entries(navItems).map(([path, name]) => (
        <Link key={path} href={path}>
          {name}
        </Link>
      ))}
    </nav>
  )
}
