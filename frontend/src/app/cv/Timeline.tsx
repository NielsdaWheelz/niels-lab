'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import rough from 'roughjs'
import { entries, gap, skills } from './data'
import { THEME_CHANGE_EVENT } from '@/lib/theme'

const CATEGORY_COLORS = {
  experience: 'var(--color-terracotta)',
  project: 'var(--color-sage)',
  education: 'var(--color-gold)',
}

const FALLBACK_LINE_COLOR = '#6f675a'

const resolveCssColor = (varName: string, fallback: string) => {
  if (typeof window === 'undefined') return fallback
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(varName)
    .trim()
  return value || fallback
}

export default function Timeline() {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const dotRefs = useRef<(HTMLDivElement | null)[]>([])
  const drawablesRef = useRef<
    ReturnType<ReturnType<typeof rough.generator>['line']>[]
  >([])
  const gapDrawableRef = useRef<ReturnType<
    ReturnType<typeof rough.generator>['line']
  > | null>(null)
  const rafRef = useRef<number | undefined>(undefined)
  const lineColorRef = useRef(FALLBACK_LINE_COLOR)
  const [visible, setVisible] = useState<Set<number>>(() => new Set())
  const [skillsVisible, setSkillsVisible] = useState(false)
  const skillsRef = useRef<HTMLDivElement>(null)

  // Generate rough.js drawables for line segments between dots
  const generateDrawables = useCallback(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return

    const containerRect = container.getBoundingClientRect()
    canvas.width = 48
    canvas.height = containerRect.height

    const generator = rough.generator()
    const drawables: typeof drawablesRef.current = []
    let gapDrawable: typeof gapDrawableRef.current = null

    const dotPositions: number[] = []
    for (let i = 0; i < dotRefs.current.length; i++) {
      const dot = dotRefs.current[i]
      if (dot) {
        const dotRect = dot.getBoundingClientRect()
        dotPositions.push(dotRect.top - containerRect.top + dotRect.height / 2)
      }
    }

    for (let i = 1; i < dotPositions.length; i++) {
      const isGapSegment = i - 1 === gap.afterIndex
      const drawable = generator.line(
        16,
        dotPositions[i - 1],
        16,
        dotPositions[i],
        {
          stroke: lineColorRef.current,
          strokeWidth: 1.5,
          roughness: 0.8,
          bowing: 0.3,
          ...(isGapSegment ? { strokeLineDash: [6, 4] } : {}),
        },
      )
      if (isGapSegment) {
        gapDrawable = drawable
      } else {
        drawables.push(drawable)
      }
    }

    // Store gap drawable at its correct position in the array
    // We need all drawables in order for progressive drawing
    const allDrawables: typeof drawablesRef.current = []
    let drawableIdx = 0
    for (let i = 1; i < dotPositions.length; i++) {
      if (i - 1 === gap.afterIndex) {
        if (gapDrawable) allDrawables.push(gapDrawable)
      } else {
        allDrawables.push(drawables[drawableIdx++])
      }
    }

    drawablesRef.current = allDrawables
    gapDrawableRef.current = gapDrawable
  }, [])

  // Draw line segments up to current scroll position
  const drawLine = useCallback(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const rc = rough.canvas(canvas)
    const containerRect = container.getBoundingClientRect()
    // How far into the container the viewport bottom reaches
    const scrollProgress =
      (window.innerHeight - containerRect.top) / containerRect.height

    const segmentCount = drawablesRef.current.length
    const visibleSegments = Math.min(
      segmentCount,
      Math.floor(scrollProgress * (segmentCount + 1)),
    )

    ctx.save()
    ctx.globalAlpha = 0.4
    for (let i = 0; i < visibleSegments; i++) {
      rc.draw(drawablesRef.current[i])
    }
    ctx.restore()
  }, [])

  // Generate drawables on mount and resize
  useEffect(() => {
    lineColorRef.current = resolveCssColor(
      '--color-text-muted',
      FALLBACK_LINE_COLOR,
    )

    // Small delay to let layout settle after ContentReveal
    const timer = setTimeout(() => {
      generateDrawables()
      drawLine()
    }, 100)

    const handleResize = () => {
      generateDrawables()
      drawLine()
    }
    window.addEventListener('resize', handleResize)

    const handleThemeChange = () => {
      lineColorRef.current = resolveCssColor(
        '--color-text-muted',
        FALLBACK_LINE_COLOR,
      )
      generateDrawables()
      drawLine()
    }
    window.addEventListener(THEME_CHANGE_EVENT, handleThemeChange)

    return () => {
      clearTimeout(timer)
      window.removeEventListener('resize', handleResize)
      window.removeEventListener(THEME_CHANGE_EVENT, handleThemeChange)
    }
  }, [generateDrawables, drawLine])

  // Scroll-linked line drawing
  useEffect(() => {
    const onScroll = () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      rafRef.current = requestAnimationFrame(drawLine)
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', onScroll)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [drawLine])

  // IntersectionObserver for entry card reveals
  useEffect(() => {
    const observer = new IntersectionObserver(
      (observerEntries) => {
        setVisible((prev) => {
          const next = new Set(prev)
          for (const entry of observerEntries) {
            if (entry.isIntersecting) {
              const idx = dotRefs.current.indexOf(
                entry.target as HTMLDivElement,
              )
              if (idx !== -1) next.add(idx)
            }
          }
          return next.size !== prev.size ? next : prev
        })
      },
      { threshold: 0.3 },
    )

    for (const dot of dotRefs.current) {
      if (dot) observer.observe(dot)
    }
    return () => observer.disconnect()
  }, [])

  // IntersectionObserver for skills section
  useEffect(() => {
    const el = skillsRef.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setSkillsVisible(true)
      },
      { threshold: 0.2 },
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return (
    <div ref={containerRef} className="timeline">
      <canvas ref={canvasRef} className="timeline-canvas" aria-hidden="true" />

      {entries.map((entry, i) => (
        <div key={i}>
          {i > 0 && i - 1 === gap.afterIndex && (
            <div className="timeline-gap">
              <p className="timeline-gap-note">{gap.note}</p>
            </div>
          )}
          <div className={`timeline-entry${visible.has(i) ? ' visible' : ''}`}>
            <div
              ref={(el) => {
                dotRefs.current[i] = el
              }}
              className={`timeline-dot${visible.has(i) ? ' visible' : ''}`}
              style={{
                borderColor: CATEGORY_COLORS[entry.category],
                backgroundColor: visible.has(i)
                  ? CATEGORY_COLORS[entry.category]
                  : 'var(--color-bg)',
              }}
            />
            <div className="timeline-card">
              <p className="timeline-title">{entry.title}</p>
              {'subtitle' in entry && entry.subtitle && (
                <p className="timeline-subtitle">{entry.subtitle}</p>
              )}
              <p className="timeline-date">{entry.date}</p>
              {'bullets' in entry && entry.bullets && (
                <ul className="timeline-bullets">
                  {entry.bullets.map((bullet, j) => (
                    <li key={j}>{bullet}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>
      ))}

      <div ref={skillsRef} className="timeline-skills">
        <p className="timeline-title" style={{ marginBottom: '1rem' }}>
          Skills
        </p>
        {Object.entries(skills).map(([group, tags]) => (
          <div key={group} className="timeline-skill-group">
            <p className="timeline-skill-label">{group}</p>
            <div className="timeline-skill-tags">
              {tags.map((tag, i) => (
                <span
                  key={tag}
                  className={`timeline-skill-tag${skillsVisible ? ' visible' : ''}`}
                  style={{
                    transitionDelay: skillsVisible ? `${i * 40}ms` : '0ms',
                  }}
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
