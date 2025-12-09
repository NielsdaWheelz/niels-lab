'use client'

import { useEffect, useRef, useState } from 'react'

interface SketchDividerProps {
  className?: string
  delay?: number
  variant?: 'wave' | 'zigzag' | 'branch' | 'dots' | 'minimal' | 'neural'
}

export function SketchDivider({ 
  className = '', 
  delay = 0,
  variant = 'wave' 
}: SketchDividerProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [hasAnimated, setHasAnimated] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

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

  // Generate organic, hand-drawn looking paths
  const paths = {
    wave: "M0 15 Q 20 5, 40 15 T 80 15 T 120 15 T 160 15 T 200 15 T 240 15 T 280 15 T 320 15",
    zigzag: "M0 20 L 15 8 L 30 20 L 45 8 L 60 20 L 75 8 L 90 20 L 105 8 L 120 20 L 135 8 L 150 20 L 165 8 L 180 20 L 195 8 L 210 20 L 225 8 L 240 20 L 255 8 L 270 20 L 285 8 L 300 20 L 315 8 L 330 20",
    branch: "M0 15 Q 30 12, 50 15 Q 70 18, 90 14 Q 110 10, 130 15 Q 150 20, 170 15 Q 190 10, 210 16 Q 230 22, 250 15 Q 270 8, 290 15 Q 310 22, 320 15",
    dots: "M10 15 L 12 15 M 40 15 L 42 15 M 70 15 L 72 15 M 100 15 L 102 15 M 130 15 L 132 15 M 160 15 L 162 15 M 190 15 L 192 15 M 220 15 L 222 15 M 250 15 L 252 15 M 280 15 L 282 15 M 310 15 L 312 15",
    // Minimal: simple sketchy line with subtle wobble
    minimal: "M0 15 Q 80 13, 160 15 Q 240 17, 320 15",
    // Neural: synapse-like with small branches
    neural: "M0 15 L 60 15 M 55 10 L 65 15 L 55 20 M 80 15 L 140 15 M 135 11 L 145 15 L 135 19 M 160 15 L 240 15 M 235 10 L 245 15 L 235 20 M 260 15 L 320 15"
  }

  // Add slight randomness to make it feel hand-drawn
  const jitter = Math.random() * 2 - 1

  return (
    <div ref={ref} className={`sketch-divider ${className}`} aria-hidden="true">
      <svg 
        viewBox="0 0 320 30" 
        fill="none" 
        preserveAspectRatio="none"
        style={{ transform: `translateY(${jitter}px)` }}
      >
        <path
          d={paths[variant]}
          stroke="var(--color-border)"
          strokeWidth="1.5"
          strokeLinecap="round"
          fill="none"
          className={`sketch-divider-path ${isVisible ? 'animate' : ''}`}
          style={{
            filter: 'url(#sketch-roughness)',
          }}
        />
        {/* Add some accent dots along the path */}
        {isVisible && variant !== 'dots' && (
          <>
            <circle 
              cx="80" 
              cy="15" 
              r="2" 
              fill="var(--color-sage)"
              className="sketch-dot"
              style={{ animationDelay: '0.4s' }}
            />
            <circle 
              cx="160" 
              cy="15" 
              r="2.5" 
              fill="var(--color-terracotta)"
              className="sketch-dot"
              style={{ animationDelay: '0.6s' }}
            />
            <circle 
              cx="240" 
              cy="15" 
              r="2" 
              fill="var(--color-gold)"
              className="sketch-dot"
              style={{ animationDelay: '0.8s' }}
            />
          </>
        )}
        {/* SVG filter for roughness */}
        <defs>
          <filter id="sketch-roughness">
            <feTurbulence type="turbulence" baseFrequency="0.05" numOctaves="2" result="noise" />
            <feDisplacementMap in="SourceGraphic" in2="noise" scale="1" xChannelSelector="R" yChannelSelector="G" />
          </filter>
        </defs>
      </svg>
    </div>
  )
}

