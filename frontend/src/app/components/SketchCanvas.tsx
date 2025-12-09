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
  const animationFrameRef = useRef<number | undefined>(undefined)
  const demoRunRef = useRef(false)
  const lastPointRef = useRef<{ x: number; y: number } | null>(null)
  const isDrawingRef = useRef(false)
  const currentBranchRef = useRef<Branch | null>(null)
  const pointCountRef = useRef(0)
  
  const [drawMode, setDrawMode] = useState(false)
  const [showHint, setShowHint] = useState(true)
  const [isMobile, setIsMobile] = useState(false)

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

  // Auto-draw demo on page load - draws complex neural network on the right side
  const runDemo = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || demoRunRef.current) return
    demoRunRef.current = true

    // Multiple starting points for a more complex pattern
    const startX = canvas.width * 0.72
    const startY = canvas.height * 0.12
    const mainBranches: Array<{ branch: Branch; points: Array<{ x: number; y: number }> }> = []

    // Create main curved trunk with organic flow
    const mainTrunkPoints: { x: number; y: number }[] = []
    let x = startX
    let y = startY
    const curveAmount = 0.4
    
    for (let i = 0; i < 60; i++) {
      const progress = i / 60
      const wave = Math.sin(progress * Math.PI * 2.5) * 12
      const curve = Math.sin(progress * Math.PI) * 25
      x += wave * curveAmount + (Math.random() - 0.5) * 2
      y += 5 + curve * 0.3 + Math.sin(progress * Math.PI * 3) * 4
      mainTrunkPoints.push({ x, y })
    }
    
    const mainBranch = startNewBranch(startX, startY)
    mainBranches.push({ branch: mainBranch, points: mainTrunkPoints })

    // Create secondary main branch that intersects
    const secondaryStartX = startX - 35
    const secondaryStartY = startY + 40
    const secondaryPoints: { x: number; y: number }[] = []
    let sx = secondaryStartX
    let sy = secondaryStartY
    
    for (let i = 0; i < 50; i++) {
      const progress = i / 50
      const spiral = Math.sin(progress * Math.PI * 4) * 15
      sx += spiral + (Math.random() - 0.5) * 3
      sy += 6 + Math.cos(progress * Math.PI * 2) * 8
      secondaryPoints.push({ x: sx, y: sy })
    }
    
    const secondaryBranch = startNewBranch(secondaryStartX, secondaryStartY)
    mainBranches.push({ branch: secondaryBranch, points: secondaryPoints })

    // Create tertiary branch from different angle
    const tertiaryStartX = startX + 20
    const tertiaryStartY = startY + 25
    const tertiaryPoints: { x: number; y: number }[] = []
    let tx = tertiaryStartX
    let ty = tertiaryStartY
    
    for (let i = 0; i < 45; i++) {
      const progress = i / 45
      const wave = Math.cos(progress * Math.PI * 3.5) * 18
      tx += wave * 0.5 + (Math.random() - 0.5) * 2.5
      ty += 7 + Math.sin(progress * Math.PI * 2.5) * 6
      tertiaryPoints.push({ x: tx, y: ty })
    }
    
    const tertiaryBranch = startNewBranch(tertiaryStartX, tertiaryStartY)
    mainBranches.push({ branch: tertiaryBranch, points: tertiaryPoints })

    // Function to create branching sub-patterns
    const createSubBranch = (
      fromX: number, 
      fromY: number, 
      angle: number, 
      length: number, 
      color: string,
      depth: number = 0
    ) => {
      if (depth > 2) return // Limit recursion depth
      
      const subBranch = startNewBranch(fromX, fromY)
      const subPoints: { x: number; y: number }[] = []
      
      for (let i = 0; i < length; i++) {
        const progress = i / length
        const curve = Math.sin(progress * Math.PI) * 8
        fromX += Math.cos(angle) * (6 + Math.random() * 3) + curve * Math.cos(angle + Math.PI / 2)
        fromY += Math.sin(angle) * (6 + Math.random() * 3) + curve * Math.sin(angle + Math.PI / 2)
        subPoints.push({ x: fromX, y: fromY })
        
        // Occasionally branch again
        if (depth < 2 && Math.random() < 0.15 && i > 3) {
          const branchAngle = angle + (Math.random() - 0.5) * Math.PI * 0.8
          createSubBranch(fromX, fromY, branchAngle, 4 + Math.floor(Math.random() * 5), color, depth + 1)
        }
      }
      
      return { branch: subBranch, points: subPoints }
    }

    // Add branches along main trunks
    mainTrunkPoints.forEach((p, i) => {
      if (i % 12 === 5 && i > 10 && i < mainTrunkPoints.length - 10) {
        const angle = Math.atan2(
          mainTrunkPoints[Math.min(i + 3, mainTrunkPoints.length - 1)].y - p.y,
          mainTrunkPoints[Math.min(i + 3, mainTrunkPoints.length - 1)].x - p.x
        )
        const branchAngle1 = angle + (Math.PI / 3 + Math.random() * 0.4)
        const branchAngle2 = angle - (Math.PI / 3 + Math.random() * 0.4)
        
        createSubBranch(p.x, p.y, branchAngle1, 6 + Math.floor(Math.random() * 6), mainBranch.color, 0)
        if (Math.random() > 0.5) {
          createSubBranch(p.x, p.y, branchAngle2, 5 + Math.floor(Math.random() * 5), mainBranch.color, 0)
        }
      }
    })

    // Add branches to secondary trunk
    secondaryPoints.forEach((p, i) => {
      if (i % 10 === 4 && i > 8 && i < secondaryPoints.length - 8) {
        const angle = Math.atan2(
          secondaryPoints[Math.min(i + 2, secondaryPoints.length - 1)].y - p.y,
          secondaryPoints[Math.min(i + 2, secondaryPoints.length - 1)].x - p.x
        )
        const branchAngle = angle + (Math.random() - 0.5) * Math.PI * 0.9
        createSubBranch(p.x, p.y, branchAngle, 4 + Math.floor(Math.random() * 5), secondaryBranch.color, 0)
      }
    })

    // Animate drawing all branches
    let branchIndex = 0
    let pointIndex = 0
    const delay = 28

    const drawNext = () => {
      if (branchIndex >= mainBranches.length) {
        // Demo finished
        return
      }

      const currentBranch = mainBranches[branchIndex]
      
      if (pointIndex >= currentBranch.points.length) {
        branchIndex++
        pointIndex = 0
        if (branchIndex < mainBranches.length) {
          setTimeout(drawNext, delay * 2)
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
      setTimeout(drawNext, delay)
    }

    // Start demo after a short delay
    setTimeout(drawNext, 600)
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
    // On mobile, enable draw mode on touch
    if (isMobile && !drawMode) {
      setDrawMode(true)
    }
    if (!drawMode) return
    
    setShowHint(false)
    isDrawingRef.current = true
    lastPointRef.current = { x: e.clientX, y: e.clientY }
    currentBranchRef.current = startNewBranch(e.clientX, e.clientY)
    pointCountRef.current = 0
  }, [drawMode, startNewBranch, isMobile])

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
    // On mobile, disable draw mode after touch ends
    if (isMobile) {
      setDrawMode(false)
    }
  }, [isMobile])

  // Detect mobile
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 640 || 'ontouchstart' in window)
    }
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  // Handle Shift key for draw mode (desktop only)
  useEffect(() => {
    if (isMobile) return

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
  }, [handlePointerUp, isMobile])

  // Add class to body when in draw mode to allow drawing through content
  useEffect(() => {
    if (drawMode) {
      document.body.classList.add('draw-mode-active')
    } else {
      document.body.classList.remove('draw-mode-active')
    }
    return () => {
      document.body.classList.remove('draw-mode-active')
    }
  }, [drawMode])

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
    canvas.addEventListener('pointercancel', handlePointerUp)

    // Run demo on load
    runDemo()

    return () => {
      window.removeEventListener('resize', resize)
      canvas.removeEventListener('pointerdown', handlePointerDown)
      canvas.removeEventListener('pointermove', handlePointerMove)
      canvas.removeEventListener('pointerup', handlePointerUp)
      canvas.removeEventListener('pointerleave', handlePointerUp)
      canvas.removeEventListener('pointercancel', handlePointerUp)
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      // Reset for React strict mode re-mount
      demoRunRef.current = false
      branchesRef.current = []
    }
    }, [draw, runDemo, handlePointerDown, handlePointerMove, handlePointerUp, isMobile])

  return (
    <>
      <canvas
        ref={canvasRef}
        className={`sketch-canvas ${drawMode ? 'draw-mode' : ''}`}
      />
      {showHint && (
        <div className="sketch-hint">
          {isMobile ? (
            <span>tap and drag to draw</span>
          ) : (
            <>
              <kbd>⇧ shift</kbd> + <span>drag to draw</span>
            </>
          )}
        </div>
      )}
    </>
  )
}
