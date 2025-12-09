'use client'

import { useRef, useState } from 'react'
import Link from 'next/link'
import { NeuralPathway } from './NeuralPathway'

const navItems = {
  '/': 'home',
  '/blog': 'blog',
  '/projects': 'projects',
  '/braindumps': 'braindumps',
  '/cv': 'cv',
}

export function Navbar() {
  const navRef = useRef<HTMLElement>(null)
  const [hoveredElement, setHoveredElement] = useState<HTMLElement | null>(null)

  return (
    <nav ref={navRef} className="nav-with-pathway">
      {Object.entries(navItems).map(([path, name]) => (
        <Link 
          key={path} 
          href={path}
          onMouseEnter={(e) => setHoveredElement(e.currentTarget)}
          onMouseLeave={() => setHoveredElement(null)}
        >
          {name}
        </Link>
      ))}
      <NeuralPathway 
        containerRef={navRef} 
        nodeSelectors={['a']}
        hoveredElement={hoveredElement}
      />
    </nav>
  )
}
