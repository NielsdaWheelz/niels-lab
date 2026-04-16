'use client'

import { useEffect, useId, useMemo, useRef, useState, ReactNode } from 'react'

interface DrawHeadingProps {
  children: ReactNode
  as?: 'h1' | 'h2' | 'h3' | 'h4'
  className?: string
  delay?: number
  underlineColor?: 'terracotta' | 'sage' | 'gold' | 'muted'
}

export function DrawHeading({
  children,
  as: Component = 'h2',
  className = '',
  delay = 0,
  underlineColor = 'terracotta',
}: DrawHeadingProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [hasAnimated, setHasAnimated] = useState(false)
  const ref = useRef<HTMLHeadingElement>(null)
  const pathId = useId()

  // simple seeded PRNG to keep SSR/CSR output identical
  const seed = useMemo(() => {
    const id = pathId
    let h = 2166136261
    for (let i = 0; i < id.length; i++) {
      h ^= id.charCodeAt(i)
      h = Math.imul(h, 16777619)
    }
    return h >>> 0
  }, [pathId])

  const underlinePath = useMemo(() => {
    let t = seed + 0x6d2b79f5
    const random = () => {
      t = Math.imul(t ^ (t >>> 15), 1 | t)
      t = t + Math.imul(t ^ (t >>> 7), 61 | t)
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }

    const points: string[] = []
    const segments = 8
    const width = 100 // percentage based

    for (let i = 0; i <= segments; i++) {
      const x = (i / segments) * width
      // Add subtle vertical wobble deterministically
      const wobble = Math.sin(i * 1.2) * 1.5 + (random() - 0.5) * 1
      const y = 3 + wobble

      if (i === 0) {
        points.push(`M ${x} ${y}`)
      } else {
        // Use quadratic curves for smoothness
        const prevX = ((i - 1) / segments) * width
        const cpX = (prevX + x) / 2
        const cpY = y + (random() - 0.5) * 2
        points.push(`Q ${cpX} ${cpY}, ${x} ${y}`)
      }
    }

    return points.join(' ')
  }, [seed])

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
      { threshold: 0.5 },
    )

    if (ref.current) {
      observer.observe(ref.current)
    }

    return () => observer.disconnect()
  }, [delay, hasAnimated])

  const colorMap = {
    terracotta: 'var(--color-terracotta)',
    sage: 'var(--color-sage)',
    gold: 'var(--color-gold)',
    muted: 'var(--color-text-muted)',
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
          d={underlinePath}
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
