'use client'

import { useEffect, useRef, useState, ReactNode, useId, useMemo } from 'react'

interface DrawHeadingProps {
  children: ReactNode
  as?: 'h1' | 'h2' | 'h3' | 'h4'
  className?: string
  delay?: number
  underlineColor?: 'terracotta' | 'sage' | 'gold' | 'muted'
}

// Simple seeded random number generator for deterministic randomness
function seededRandom(seed: number) {
  let value = seed
  return () => {
    value = (value * 9301 + 49297) % 233280
    return value / 233280
  }
}

// Extract text content from ReactNode deterministically
function extractText(node: ReactNode): string {
  if (node == null) {
    return ''
  }
  if (typeof node === 'string' || typeof node === 'number') {
    return String(node).trim()
  }
  if (typeof node === 'boolean') {
    return ''
  }
  if (Array.isArray(node)) {
    return node.map(extractText).filter(Boolean).join('').trim()
  }
  if (typeof node === 'object') {
    // Handle React elements
    if ('props' in node && node.props && typeof node.props === 'object') {
      if ('children' in node.props) {
        return extractText(node.props.children as ReactNode)
      }
    }
  }
  return ''
}

// Generate a hash from a string for seeding
function hashString(str: string): number {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash | 0 // Convert to 32-bit integer
  }
  return Math.abs(hash)
}

export function DrawHeading({ 
  children, 
  as: Component = 'h2',
  className = '',
  delay = 0,
  underlineColor = 'terracotta'
}: DrawHeadingProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [hasAnimated, setHasAnimated] = useState(false)
  const ref = useRef<HTMLHeadingElement>(null)
  const pathId = useId()

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasAnimated) {
          setTimeout(() => {
            setIsVisible(true)
            setHasAnimated(true)
          }, delay)
        }
      },
      { threshold: 0.5 }
    )

    if (ref.current) {
      observer.observe(ref.current)
    }

    return () => observer.disconnect()
  }, [delay, hasAnimated])

  // Extract text content once for use in memoization
  const textContent = useMemo(() => extractText(children), [children])
  
  // Generate a slightly wobbly line path deterministically
  const pathData = useMemo(() => {
    // Create a deterministic seed based on text content and props
    const seedString = `${textContent}-${underlineColor}-${delay}`
    const seed = hashString(seedString)
    const random = seededRandom(seed)
    
    // Helper to round to fixed precision for consistency
    const round = (n: number, precision = 10) => {
      return Math.round(n * precision) / precision
    }
    
    const points: string[] = []
    const segments = 8
    const width = 100 // percentage based
    
    for (let i = 0; i <= segments; i++) {
      const x = round((i / segments) * width)
      // Add subtle vertical wobble using seeded random
      const wobble = Math.sin(i * 1.2) * 1.5 + (random() - 0.5) * 1
      const y = round(3 + wobble)
      
      if (i === 0) {
        points.push(`M ${x} ${y}`)
      } else {
        // Use quadratic curves for smoothness
        const prevX = round(((i - 1) / segments) * width)
        const cpX = round((prevX + x) / 2)
        const cpY = round(y + (random() - 0.5) * 2)
        points.push(`Q ${cpX} ${cpY}, ${x} ${y}`)
      }
    }
    
    return points.join(' ')
  }, [textContent, underlineColor, delay])

  const colorMap = {
    terracotta: 'var(--color-terracotta)',
    sage: 'var(--color-sage)',
    gold: 'var(--color-gold)',
    muted: 'var(--color-text-muted)'
  }

  return (
    <Component ref={ref} className={`draw-heading ${className}`}>
      <span className="draw-heading-text">{children}</span>
      <svg 
        className={`draw-heading-underline ${isVisible ? 'animate' : ''}`}
        viewBox="0 0 100 8" 
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <path
          id={pathId}
          d={pathData}
          stroke={colorMap[underlineColor]}
          strokeWidth="1.5"
          strokeLinecap="round"
          fill="none"
          className="draw-underline-path"
        />
        {/* Small accent dot at the end */}
        <circle
          cx="100"
          cy="4"
          r="1.5"
          fill={colorMap[underlineColor]}
          className="draw-underline-dot"
          opacity="0"
        />
      </svg>
    </Component>
  )
}

