'use client'

import { useRef, useState, ReactNode } from 'react'
import { NeuralPathway } from './NeuralPathway'

interface NeuralSectionProps {
  children: ReactNode
  className?: string
  nodeSelectors?: string[]
}

export function NeuralSection({ 
  children, 
  className = '',
  nodeSelectors = ['h2', 'a', 'li']
}: NeuralSectionProps) {
  const sectionRef = useRef<HTMLElement>(null)
  const [hoveredElement, setHoveredElement] = useState<HTMLElement | null>(null)

  // Add hover handlers to h2 elements within this section
  const handleMouseEnter = (e: React.MouseEvent<HTMLElement>) => {
    if (e.target instanceof HTMLElement && e.target.tagName === 'H2') {
      setHoveredElement(e.target)
    }
  }

  const handleMouseLeave = () => {
    setHoveredElement(null)
  }

  return (
    <section 
      ref={sectionRef} 
      className={`neural-section ${className}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {children}
      <NeuralPathway 
        containerRef={sectionRef} 
        nodeSelectors={nodeSelectors}
        hoveredElement={hoveredElement}
      />
    </section>
  )
}
