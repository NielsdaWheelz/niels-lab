'use client'

import { useEffect, useRef } from 'react'
import rough from 'roughjs'

interface RoughCardBorderProps {
  className?: string
}

export function RoughCardBorder({ className = '' }: RoughCardBorderProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapperRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const wrapper = wrapperRef.current
    if (!canvas || !wrapper) return

    // Get the parent container (the .project-image-card)
    const parent = wrapper.parentElement
    if (!parent) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const updateCanvas = () => {
      const rect = parent.getBoundingClientRect()
      const width = Math.round(rect.width)
      const height = Math.round(rect.height)

      if (width === 0 || height === 0) return

      // Set canvas size to match the card (using device pixel ratio for crisp rendering)
      const dpr = window.devicePixelRatio || 1
      canvas.width = width * dpr
      canvas.height = height * dpr
      ctx.scale(dpr, dpr)

      // Clear previous drawing
      ctx.clearRect(0, 0, width, height)

      const rc = rough.canvas(canvas)

      // Draw a sketchy rectangle border
      rc.rectangle(2, 2, width - 4, height - 4, {
        stroke: 'var(--color-border)',
        strokeWidth: 1.5,
        roughness: 1.2,
        bowing: 2,
        fill: 'transparent',
        fillStyle: 'solid',
      })

      // Add some extra sketchy lines for character
      const randomOffset = () => (Math.random() - 0.5) * 3

      // Top edge variation
      rc.line(4 + randomOffset(), 3, width - 4 + randomOffset(), 3, {
        stroke: 'var(--color-border)',
        strokeWidth: 1,
        roughness: 1.5,
        opacity: 0.6,
      })

      // Bottom edge variation
      rc.line(4 + randomOffset(), height - 3, width - 4 + randomOffset(), height - 3, {
        stroke: 'var(--color-border)',
        strokeWidth: 1,
        roughness: 1.5,
        opacity: 0.6,
      })

      // Left edge variation
      rc.line(3, 4 + randomOffset(), 3, height - 4 + randomOffset(), {
        stroke: 'var(--color-border)',
        strokeWidth: 1,
        roughness: 1.5,
        opacity: 0.6,
      })

      // Right edge variation
      rc.line(width - 3, 4 + randomOffset(), width - 3, height - 4 + randomOffset(), {
        stroke: 'var(--color-border)',
        strokeWidth: 1,
        roughness: 1.5,
        opacity: 0.6,
      })
    }

    // Initial draw (with a small delay to ensure parent is sized)
    const timeoutId = setTimeout(updateCanvas, 0)

    // Handle resize
    const resizeObserver = new ResizeObserver(updateCanvas)
    resizeObserver.observe(parent)

    return () => {
      clearTimeout(timeoutId)
      resizeObserver.disconnect()
    }
  }, [])

  return (
    <div 
      ref={wrapperRef} 
      style={{ 
        position: 'absolute', 
        top: 0, 
        left: 0, 
        width: '100%', 
        height: '100%', 
        pointerEvents: 'none', 
        zIndex: 2 
      }}
    >
      <canvas
        ref={canvasRef}
        className={`rough-card-border ${className}`}
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
        }}
        aria-hidden="true"
      />
    </div>
  )
}

