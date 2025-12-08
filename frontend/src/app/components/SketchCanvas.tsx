'use client'

import { useEffect, useRef, useCallback, useState } from 'react'
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

const COLORS = ['#E07A5F', '#81B29A', '#F2CC8F', '#3D3D3D']
const FADE_START = 3000
const FADE_DURATION = 4000
const POINT_INTERVAL = 8
const BRANCH_CHANCE = 0.02

export function SketchCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const branchesRef = useRef<Branch[]>([])
  const animationFrameRef = useRef<number>()
  const demoRunRef = useRef(false)
  const lastPointRef = useRef<{ x: number; y: number } | null>(null)
  const isDrawingRef = useRef(false)
  const currentBranchRef = useRef<Branch | null>(null)
  const pointCountRef = useRef(0)
  
  const [drawMode, setDrawMode] = useState(false)
  const [showHint, setShowHint] = useState(false)

  const getRandomColor = () => COLORS[Math.floor(Math.random() * COLORS.length)]

  const startNewBranch = useCallback((x: number, y: number, color?: string) => {
    const branch: Branch = {
      points: [{ x, y, age: Date.now() }],
      color: color || getRandomColor(),
      createdAt: Date.now(),
    }
    branchesRef.current.push(branch)
    return branch
  }, [])

  // Auto-draw demo on page load - draws on the right side
  const runDemo = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || demoRunRef.current) return
    demoRunRef.current = true

    // Start from the right side of the screen
    const startX = canvas.width * 0.7
    const startY = canvas.height * 0.15

    const branch = startNewBranch(startX, startY)
    
    // Create a flowing path that goes down and slightly left
    const points: { x: number; y: number }[] = []
    let x = startX
    let y = startY
    
    for (let i = 0; i < 40; i++) {
      x += Math.sin(i * 0.2) * 8 + (Math.random() - 0.6) * 3
      y += 6 + Math.random() * 3
      points.push({ x, y })
    }

    // Animate drawing the points
    let pointIndex = 0
    const drawNextPoint = () => {
      if (pointIndex >= points.length) return
      
      const p = points[pointIndex]
      branch.points.push({
        x: p.x,
        y: p.y,
        age: Date.now(),
      })

      // Spawn branches at specific points - going outward
      if (pointIndex === 10 || pointIndex === 22 || pointIndex === 32) {
        const direction = pointIndex === 22 ? -1 : 1
        const angle = direction * (Math.PI / 3 + Math.random() * 0.3)
        const newBranch = startNewBranch(p.x, p.y, branch.color)
        
        let bx = p.x
        let by = p.y
        const branchLength = 5 + Math.floor(Math.random() * 4)
        for (let j = 0; j < branchLength; j++) {
          bx += Math.cos(angle) * (7 + Math.random() * 4)
          by += Math.sin(angle) * (5 + Math.random() * 3)
          newBranch.points.push({
            x: bx,
            y: by,
            age: Date.now() + j * 40,
          })
        }
      }

      pointIndex++
      if (pointIndex < points.length) {
        setTimeout(drawNextPoint, 35)
      } else {
        // Demo finished, show hint after a short delay
        setTimeout(() => setShowHint(true), 1500)
      }
    }

    // Start demo after a short delay
    setTimeout(drawNextPoint, 600)
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
      const lastPointAge = branch.points.length > 0 
        ? now - branch.points[branch.points.length - 1].age 
        : now - branch.createdAt

      // Remove if fully faded
      if (lastPointAge > FADE_START + FADE_DURATION) return false

      // Calculate opacity based on age of the last point
      let opacity = 1
      if (lastPointAge > FADE_START) {
        opacity = Math.max(0, 1 - (lastPointAge - FADE_START) / FADE_DURATION)
      }

      ctx.save()
      ctx.globalAlpha = opacity * 0.5

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

      // Draw subtle node at the end
      if (branch.points.length > 0) {
        const lastPoint = branch.points[branch.points.length - 1]
        ctx.globalAlpha = opacity * 0.25
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
        
        // Only connect points from different branches
        if (b1 === b2) continue
        
        const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y)
        const CONNECTION_DISTANCE = 120
        
        if (dist < CONNECTION_DISTANCE && dist > 20) {
          const connectionOpacity = (1 - dist / CONNECTION_DISTANCE) * 0.15
          ctx.globalAlpha = connectionOpacity
          
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
    if (!drawMode) return
    
    setShowHint(false)
    isDrawingRef.current = true
    lastPointRef.current = { x: e.clientX, y: e.clientY }
    currentBranchRef.current = startNewBranch(e.clientX, e.clientY)
    pointCountRef.current = 0
  }, [drawMode, startNewBranch])

  const handlePointerMove = useCallback((e: PointerEvent) => {
    if (!isDrawingRef.current || !lastPointRef.current || !currentBranchRef.current) return

    const dx = e.clientX - lastPointRef.current.x
    const dy = e.clientY - lastPointRef.current.y
    const distance = Math.hypot(dx, dy)

    if (distance > POINT_INTERVAL) {
      currentBranchRef.current.points.push({
        x: e.clientX,
        y: e.clientY,
        age: Date.now(),
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

  // Handle Shift key for draw mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Shift' && !e.repeat) {
        setDrawMode(true)
      }
    }

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'Shift') {
        setDrawMode(false)
        handlePointerUp()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [handlePointerUp])

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
      // Reset for React strict mode re-mount
      demoRunRef.current = false
      branchesRef.current = []
    }
  }, [draw, runDemo, handlePointerDown, handlePointerMove, handlePointerUp])

  return (
    <>
      <canvas
        ref={canvasRef}
        className={`sketch-canvas ${drawMode ? 'draw-mode' : ''}`}
      />
      {showHint && (
        <div className="sketch-hint">
          <kbd>⇧ shift</kbd> + drag to trace
        </div>
      )}
    </>
  )
}
