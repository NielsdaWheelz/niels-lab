'use client'

import { useEffect, useRef, useCallback } from 'react'
import rough from 'roughjs'

interface Point {
  x: number
  y: number
  age: number
}

interface Branch {
  points: Point[]
  color: string
  createdAt: number
}

interface ProjectCardSketchProps {
  className?: string
}

const FADE_START = 4000
const FADE_DURATION = 3000
const DRAW_DELAY = 25

export function ProjectCardSketch({ className = '' }: ProjectCardSketchProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const branchesRef = useRef<Branch[]>([])
  const animationFrameRef = useRef<number | null>(null)
  const demoRunRef = useRef(false)
  const hasStartedRef = useRef(false)

  const getRandomColor = () => {
    // Get computed color values from CSS variables
    const style = getComputedStyle(document.documentElement)
    const colors = [
      style.getPropertyValue('--color-terracotta').trim() || '#1A7080',
      style.getPropertyValue('--color-sage').trim() || '#7A8B5C',
      style.getPropertyValue('--color-gold').trim() || '#D4A84A',
      style.getPropertyValue('--color-text').trim() || '#3D3D3D',
    ]
    return colors[Math.floor(Math.random() * colors.length)]
  }

  const startNewBranch = useCallback((x: number, y: number, color?: string) => {
    const branch: Branch = {
      points: [{ x, y, age: Date.now() }],
      color: color || getRandomColor(),
      createdAt: Date.now(),
    }
    branchesRef.current.push(branch)
    return branch
  }, [])

  // Auto-draw complex pattern when card becomes visible
  const runDemo = useCallback(() => {
    const canvas = canvasRef.current
    const wrapper = wrapperRef.current
    if (!canvas || !wrapper || demoRunRef.current || hasStartedRef.current) return

    const parent = wrapper.parentElement
    if (!parent) return

    hasStartedRef.current = true
    demoRunRef.current = true

    const rect = parent.getBoundingClientRect()
    const width = rect.width
    const height = rect.height

    if (width === 0 || height === 0) return

    // Multiple starting points for complexity
    const mainBranches: Array<{ branch: Branch; points: Array<{ x: number; y: number }> }> = []

    // Main trunk - curved organic flow
    const startX = width * 0.2
    const startY = height * 0.3
    const mainTrunkPoints: { x: number; y: number }[] = []
    let x = startX
    let y = startY

    for (let i = 0; i < 40; i++) {
      const progress = i / 40
      const wave = Math.sin(progress * Math.PI * 3) * 15
      const curve = Math.sin(progress * Math.PI) * 20
      x += wave * 0.3 + (Math.random() - 0.5) * 1.5
      y += height / 45 + curve * 0.25 + Math.sin(progress * Math.PI * 4) * 3
      mainTrunkPoints.push({ x: Math.max(5, Math.min(width - 5, x)), y: Math.max(5, Math.min(height - 5, y)) })
    }

    const mainBranch = startNewBranch(mainTrunkPoints[0].x, mainTrunkPoints[0].y)
    mainBranches.push({ branch: mainBranch, points: mainTrunkPoints })

    // Secondary branch from different angle
    const secondaryStartX = width * 0.7
    const secondaryStartY = height * 0.2
    const secondaryPoints: { x: number; y: number }[] = []
    let sx = secondaryStartX
    let sy = secondaryStartY

    for (let i = 0; i < 35; i++) {
      const progress = i / 35
      const spiral = Math.sin(progress * Math.PI * 3.5) * 12
      sx += spiral * 0.4 + (Math.random() - 0.5) * 2
      sy += height / 40 + Math.cos(progress * Math.PI * 2.5) * 5
      secondaryPoints.push({ x: Math.max(5, Math.min(width - 5, sx)), y: Math.max(5, Math.min(height - 5, sy)) })
    }

    const secondaryBranch = startNewBranch(secondaryPoints[0].x, secondaryPoints[0].y)
    mainBranches.push({ branch: secondaryBranch, points: secondaryPoints })

    // Third branch for more complexity
    const tertiaryStartX = width * 0.5
    const tertiaryStartY = height * 0.15
    const tertiaryPoints: { x: number; y: number }[] = []
    let tx = tertiaryStartX
    let ty = tertiaryStartY

    for (let i = 0; i < 30; i++) {
      const progress = i / 30
      const wave = Math.cos(progress * Math.PI * 2.8) * 14
      tx += wave * 0.35 + (Math.random() - 0.5) * 2
      ty += height / 35 + Math.sin(progress * Math.PI * 2.2) * 4
      tertiaryPoints.push({ x: Math.max(5, Math.min(width - 5, tx)), y: Math.max(5, Math.min(height - 5, ty)) })
    }

    const tertiaryBranch = startNewBranch(tertiaryPoints[0].x, tertiaryPoints[0].y)
    mainBranches.push({ branch: tertiaryBranch, points: tertiaryPoints })

    // Function to create sub-branches
    const createSubBranch = (
      fromX: number,
      fromY: number,
      angle: number,
      length: number,
      color: string,
      depth: number = 0
    ) => {
      if (depth > 2) return

      const subBranch = startNewBranch(fromX, fromY)
      const subPoints: { x: number; y: number }[] = []
      let fx = fromX
      let fy = fromY

      for (let i = 0; i < length; i++) {
        const progress = i / length
        const curve = Math.sin(progress * Math.PI) * 6
        fx += Math.cos(angle) * (4 + Math.random() * 2) + curve * Math.cos(angle + Math.PI / 2) * 0.3
        fy += Math.sin(angle) * (4 + Math.random() * 2) + curve * Math.sin(angle + Math.PI / 2) * 0.3
        subPoints.push({
          x: Math.max(5, Math.min(width - 5, fx)),
          y: Math.max(5, Math.min(height - 5, fy)),
        })

        if (depth < 2 && Math.random() < 0.12 && i > 2) {
          const branchAngle = angle + (Math.random() - 0.5) * Math.PI * 0.7
          createSubBranch(fx, fy, branchAngle, 3 + Math.floor(Math.random() * 4), color, depth + 1)
        }
      }

      return { branch: subBranch, points: subPoints }
    }

    // Add branches along main trunks
    mainTrunkPoints.forEach((p, i) => {
      if (i % 8 === 4 && i > 5 && i < mainTrunkPoints.length - 5) {
        const nextIdx = Math.min(i + 2, mainTrunkPoints.length - 1)
        const angle = Math.atan2(mainTrunkPoints[nextIdx].y - p.y, mainTrunkPoints[nextIdx].x - p.x)
        const branchAngle1 = angle + (Math.PI / 2.5 + Math.random() * 0.3)
        const branchAngle2 = angle - (Math.PI / 2.5 + Math.random() * 0.3)

        createSubBranch(p.x, p.y, branchAngle1, 5 + Math.floor(Math.random() * 5), mainBranch.color, 0)
        if (Math.random() > 0.4) {
          createSubBranch(p.x, p.y, branchAngle2, 4 + Math.floor(Math.random() * 4), mainBranch.color, 0)
        }
      }
    })

    secondaryPoints.forEach((p, i) => {
      if (i % 7 === 3 && i > 4 && i < secondaryPoints.length - 4) {
        const nextIdx = Math.min(i + 2, secondaryPoints.length - 1)
        const angle = Math.atan2(secondaryPoints[nextIdx].y - p.y, secondaryPoints[nextIdx].x - p.x)
        const branchAngle = angle + (Math.random() - 0.5) * Math.PI * 0.8
        createSubBranch(p.x, p.y, branchAngle, 4 + Math.floor(Math.random() * 4), secondaryBranch.color, 0)
      }
    })

    // Animate drawing
    let branchIndex = 0
    let pointIndex = 0

    const drawNext = () => {
      if (branchIndex >= mainBranches.length) return

      const currentBranch = mainBranches[branchIndex]

      if (pointIndex >= currentBranch.points.length) {
        branchIndex++
        pointIndex = 0
        if (branchIndex < mainBranches.length) {
          setTimeout(drawNext, DRAW_DELAY * 1.5)
        }
        return
      }

      const p = currentBranch.points[pointIndex]
      currentBranch.branch.points.push({
        x: p.x,
        y: p.y,
        age: Date.now(),
      })

      pointIndex++
      setTimeout(drawNext, DRAW_DELAY)
    }

    setTimeout(drawNext, 300)
  }, [startNewBranch])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const wrapper = wrapperRef.current
    if (!wrapper) return

    const parent = wrapper.parentElement
    if (!parent) return

    const rect = parent.getBoundingClientRect()
    const width = Math.round(rect.width)
    const height = Math.round(rect.height)

    if (width === 0 || height === 0) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)

    const rc = rough.canvas(canvas)
    const now = Date.now()

    // Draw background (subtle)
    ctx.fillStyle = 'var(--color-bg-alt)'
    ctx.fillRect(0, 0, width, height)

    // Draw border with rough.js
    rc.rectangle(2, 2, width - 4, height - 4, {
      stroke: 'var(--color-border)',
      strokeWidth: 1.5,
      roughness: 1.2,
      bowing: 2,
      fill: 'transparent',
    })

    // Filter and draw branches
    branchesRef.current = branchesRef.current.filter(branch => {
      const lastPointAge = branch.points.length > 0
        ? now - branch.points[branch.points.length - 1].age
        : now - branch.createdAt

      if (lastPointAge > FADE_START + FADE_DURATION) return false

      let opacity = 1
      if (lastPointAge > FADE_START) {
        opacity = Math.max(0, 1 - (lastPointAge - FADE_START) / FADE_DURATION)
      }

      ctx.save()
      ctx.globalAlpha = opacity * 0.6

      if (branch.points.length > 1) {
        for (let i = 1; i < branch.points.length; i++) {
          const p1 = branch.points[i - 1]
          const p2 = branch.points[i]

          rc.line(p1.x, p1.y, p2.x, p2.y, {
            stroke: branch.color,
            strokeWidth: 1.2,
            roughness: 0.7,
            bowing: 0.4,
          })
        }
      }

      if (branch.points.length > 0) {
        const lastPoint = branch.points[branch.points.length - 1]
        ctx.globalAlpha = opacity * 0.3
        rc.circle(lastPoint.x, lastPoint.y, 3, {
          stroke: branch.color,
          strokeWidth: 0.8,
          roughness: 0.8,
          fill: branch.color,
          fillStyle: 'solid',
        })
      }

      ctx.restore()
      return true
    })

    // Draw connections between nearby points
    const allPoints: { point: Point; branch: Branch }[] = []
    branchesRef.current.forEach(branch => {
      branch.points.forEach(point => {
        allPoints.push({ point, branch })
      })
    })

    ctx.save()
    for (let i = 0; i < allPoints.length; i++) {
      for (let j = i + 1; j < allPoints.length; j++) {
        const { point: p1, branch: b1 } = allPoints[i]
        const { point: p2, branch: b2 } = allPoints[j]

        if (b1 === b2) continue

        const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y)
        const CONNECTION_DISTANCE = Math.min(width, height) * 0.6

        if (dist < CONNECTION_DISTANCE && dist > 15) {
          const connectionOpacity = (1 - dist / CONNECTION_DISTANCE) * 0.12
          ctx.globalAlpha = connectionOpacity

          rc.line(p1.x, p1.y, p2.x, p2.y, {
            stroke: 'var(--color-text)',
            strokeWidth: 0.5,
            roughness: 1.2,
            bowing: 0.8,
          })
        }
      }
    }
    ctx.restore()

    animationFrameRef.current = requestAnimationFrame(draw)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    const wrapper = wrapperRef.current
    if (!canvas || !wrapper) return

    const parent = wrapper.parentElement
    if (!parent) return

    // Intersection Observer to start animation when card is visible
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !hasStartedRef.current) {
            runDemo()
          }
        })
      },
      { threshold: 0.1 }
    )

    observer.observe(parent)
    animationFrameRef.current = requestAnimationFrame(draw)

    return () => {
      observer.disconnect()
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      branchesRef.current = []
      hasStartedRef.current = false
      demoRunRef.current = false
    }
  }, [draw, runDemo])

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
        zIndex: 1,
      }}
    >
      <canvas
        ref={canvasRef}
        className={`project-card-sketch ${className}`}
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



