'use client'

import { useEffect, useRef, useCallback, useState } from 'react'
import rough from 'roughjs'

interface Point {
  x: number
  y: number
  age: number
  connections: number[]
}

interface Branch {
  points: Point[]
  color: string
  createdAt: number
}

const COLORS = ['#E07A5F', '#81B29A', '#F2CC8F', '#3D3D3D']
const FADE_START = 2000
const FADE_DURATION = 3000
const CONNECTION_DISTANCE = 120
const BRANCH_CHANCE = 0.02
const POINT_INTERVAL = 8

export function SketchCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const branchesRef = useRef<Branch[]>([])
  const animationFrameRef = useRef<number>()
  const lastPointRef = useRef<{ x: number; y: number } | null>(null)
  const isDrawingRef = useRef(false)
  const currentBranchRef = useRef<Branch | null>(null)
  const pointCountRef = useRef(0)
  const hasInteractedRef = useRef(false)
  const [showHint, setShowHint] = useState(true)

  const getRandomColor = () => COLORS[Math.floor(Math.random() * COLORS.length)]

  const startNewBranch = useCallback((x: number, y: number, color?: string) => {
    const branch: Branch = {
      points: [{ x, y, age: Date.now(), connections: [] }],
      color: color || getRandomColor(),
      createdAt: Date.now(),
    }
    branchesRef.current.push(branch)
    return branch
  }, [])

  // Auto-draw demo on page load
  const runDemo = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || hasInteractedRef.current) return

    const centerX = canvas.width * 0.5
    const centerY = canvas.height * 0.35

    const branch = startNewBranch(centerX - 150, centerY)
    
    // Create a flowing path
    const points: { x: number; y: number }[] = []
    let x = centerX - 150
    let y = centerY
    
    for (let i = 0; i < 40; i++) {
      x += 8 + Math.random() * 4
      y += Math.sin(i * 0.3) * 8 + (Math.random() - 0.5) * 6
      points.push({ x, y })
    }

    // Animate drawing the points
    let pointIndex = 0
    const drawNextPoint = () => {
      if (pointIndex >= points.length || hasInteractedRef.current) return
      
      const p = points[pointIndex]
      branch.points.push({
        x: p.x,
        y: p.y,
        age: Date.now(),
        connections: [],
      })

      // Spawn a branch occasionally
      if (pointIndex === 15 || pointIndex === 28) {
        const angle = (pointIndex === 15 ? -1 : 1) * (Math.PI / 4 + Math.random() * 0.3)
        const newBranch = startNewBranch(p.x, p.y, branch.color)
        
        let bx = p.x
        let by = p.y
        for (let j = 0; j < 8; j++) {
          bx += Math.cos(angle) * (10 + Math.random() * 5)
          by += Math.sin(angle) * (10 + Math.random() * 5)
          newBranch.points.push({
            x: bx,
            y: by,
            age: Date.now() + j * 30,
            connections: [],
          })
        }
      }

      pointIndex++
      setTimeout(drawNextPoint, 25)
    }

    // Start demo after a short delay
    setTimeout(drawNextPoint, 800)
  }, [startNewBranch])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rc = rough.canvas(canvas)
    const now = Date.now()

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Filter out old branches and draw remaining
    branchesRef.current = branchesRef.current.filter(branch => {
      const branchAge = now - branch.createdAt
      const lastPointAge = branch.points.length > 0 
        ? now - branch.points[branch.points.length - 1].age 
        : branchAge

      // Remove if fully faded
      if (lastPointAge > FADE_START + FADE_DURATION) return false

      // Calculate opacity based on age of the last point
      let opacity = 1
      if (lastPointAge > FADE_START) {
        opacity = Math.max(0, 1 - (lastPointAge - FADE_START) / FADE_DURATION)
      }

      ctx.save()
      ctx.globalAlpha = opacity * 0.6

      // Draw the path
      if (branch.points.length > 1) {
        for (let i = 1; i < branch.points.length; i++) {
          const p1 = branch.points[i - 1]
          const p2 = branch.points[i]
          
          rc.line(p1.x, p1.y, p2.x, p2.y, {
            stroke: branch.color,
            strokeWidth: 1.5,
            roughness: 0.8,
            bowing: 0.5,
          })
        }
      }

      // Draw subtle nodes at branch points
      if (branch.points.length > 0) {
        const lastPoint = branch.points[branch.points.length - 1]
        ctx.globalAlpha = opacity * 0.3
        rc.circle(lastPoint.x, lastPoint.y, 4, {
          stroke: branch.color,
          strokeWidth: 1,
          roughness: 1,
          fill: branch.color,
          fillStyle: 'solid',
        })
      }

      ctx.restore()
      return true
    })

    // Draw connections between nearby points from different branches
    ctx.save()
    ctx.globalAlpha = 0.15
    
    const allPoints: { point: Point; branch: Branch }[] = []
    branchesRef.current.forEach(branch => {
      branch.points.forEach(point => {
        allPoints.push({ point, branch })
      })
    })

    for (let i = 0; i < allPoints.length; i++) {
      for (let j = i + 1; j < allPoints.length; j++) {
        const { point: p1, branch: b1 } = allPoints[i]
        const { point: p2, branch: b2 } = allPoints[j]
        
        // Only connect points from different branches
        if (b1 === b2) continue
        
        const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y)
        if (dist < CONNECTION_DISTANCE && dist > 20) {
          const connectionOpacity = 1 - (dist / CONNECTION_DISTANCE)
          ctx.globalAlpha = connectionOpacity * 0.1
          
          rc.line(p1.x, p1.y, p2.x, p2.y, {
            stroke: '#3D3D3D',
            strokeWidth: 0.5,
            roughness: 1.5,
            bowing: 1,
          })
        }
      }
    }
    
    ctx.restore()

    animationFrameRef.current = requestAnimationFrame(draw)
  }, [])

  const handlePointerDown = useCallback((e: PointerEvent) => {
    if (!hasInteractedRef.current) {
      hasInteractedRef.current = true
      setShowHint(false)
    }
    
    isDrawingRef.current = true
    lastPointRef.current = { x: e.clientX, y: e.clientY }
    currentBranchRef.current = startNewBranch(e.clientX, e.clientY)
    pointCountRef.current = 0
  }, [startNewBranch])

  const handlePointerMove = useCallback((e: PointerEvent) => {
    if (!isDrawingRef.current || !lastPointRef.current || !currentBranchRef.current) return

    const dx = e.clientX - lastPointRef.current.x
    const dy = e.clientY - lastPointRef.current.y
    const distance = Math.hypot(dx, dy)

    if (distance > POINT_INTERVAL) {
      // Add point to current branch
      currentBranchRef.current.points.push({
        x: e.clientX,
        y: e.clientY,
        age: Date.now(),
        connections: [],
      })

      pointCountRef.current++

      // Occasionally spawn a new branch
      if (Math.random() < BRANCH_CHANCE && pointCountRef.current > 10) {
        const angle = Math.atan2(dy, dx) + (Math.random() - 0.5) * Math.PI * 0.8
        const branchLength = 30 + Math.random() * 50
        const branchX = e.clientX + Math.cos(angle) * branchLength
        const branchY = e.clientY + Math.sin(angle) * branchLength
        
        const newBranch = startNewBranch(e.clientX, e.clientY, currentBranchRef.current.color)
        newBranch.points.push({
          x: branchX,
          y: branchY,
          age: Date.now(),
          connections: [],
        })
      }

      lastPointRef.current = { x: e.clientX, y: e.clientY }
    }
  }, [startNewBranch])

  const handlePointerUp = useCallback(() => {
    isDrawingRef.current = false
    lastPointRef.current = null
    currentBranchRef.current = null
    pointCountRef.current = 0
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    resize()
    window.addEventListener('resize', resize)

    animationFrameRef.current = requestAnimationFrame(draw)

    canvas.addEventListener('pointerdown', handlePointerDown)
    canvas.addEventListener('pointermove', handlePointerMove)
    canvas.addEventListener('pointerup', handlePointerUp)
    canvas.addEventListener('pointerleave', handlePointerUp)

    // Run demo on load
    runDemo()

    return () => {
      window.removeEventListener('resize', resize)
      canvas.removeEventListener('pointerdown', handlePointerDown)
      canvas.removeEventListener('pointermove', handlePointerMove)
      canvas.removeEventListener('pointerup', handlePointerUp)
      canvas.removeEventListener('pointerleave', handlePointerUp)
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [draw, handlePointerDown, handlePointerMove, handlePointerUp, runDemo])

  return (
    <>
      <canvas
        ref={canvasRef}
        className="sketch-canvas interactive"
        style={{ cursor: 'crosshair' }}
      />
      {showHint && (
        <div className="sketch-hint">
          <span>drag to trace</span>
        </div>
      )}
    </>
  )
}
