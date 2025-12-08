'use client'

import { useRef, useState, ReactNode, useEffect } from 'react'
import { NeuralPathway } from './NeuralPathway'

interface NeuralHeadingSectionProps {
  children: ReactNode
  className?: string
}

export function NeuralHeadingSection({ 
  children, 
  className = ''
}: NeuralHeadingSectionProps) {
  const headingRef = useRef<HTMLDivElement>(null)
  const [hoveredElement, setHoveredElement] = useState<HTMLElement | null>(null)

  useEffect(() => {
    const container = headingRef.current
    if (!container) return

    const h2 = container.querySelector('h2')
    if (!h2) return

    const handleMouseEnter = () => {
      setHoveredElement(h2 as HTMLElement)
    }

    const handleMouseLeave = () => {
      setHoveredElement(null)
    }

    h2.addEventListener('mouseenter', handleMouseEnter)
    h2.addEventListener('mouseleave', handleMouseLeave)

    return () => {
      h2.removeEventListener('mouseenter', handleMouseEnter)
      h2.removeEventListener('mouseleave', handleMouseLeave)
    }
  }, [])

  return (
    <div ref={headingRef} className={`neural-heading-section ${className}`}>
      {children}
      <NeuralPathway 
        containerRef={headingRef} 
        nodeSelectors={['h2']}
        hoveredElement={hoveredElement}
      />
    </div>
  )
}

