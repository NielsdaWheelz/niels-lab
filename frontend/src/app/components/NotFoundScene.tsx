'use client'

import { useEffect, useRef } from 'react'
import rough from 'roughjs'
import { THEME_CHANGE_EVENT } from '@/lib/theme'
import styles from './NotFoundScene.module.css'

type RoughCanvas = ReturnType<typeof rough.canvas>
type Pt = { x: number; y: number }
type ColorKey = 'text' | 'muted' | 'terracotta' | 'sage' | 'gold'
type Colors = Record<ColorKey, string>

interface Spec {
  jag: number[]
  wander: number[]
}

interface StrokeDef {
  points: Pt[]
  colorKey: ColorKey
  strokeWidth: number
  roughness: number
  bowing: number
  seed: number
  dash?: number[]
  alpha: number
  start: number
  end: number
  rotate?: { cx: number; cy: number; angle: number }
}

const TOTAL_MS = 2600

// small hand-sketched scene: a torn page trailing a dashed, wandering path
// that leads to an "X marks the spot". Drawn progressively on mount, like
// the rest of the site's sketch animations.

function createSpec(): Spec {
  const rand = (min: number, max: number) => min + Math.random() * (max - min)
  const jag = Array.from({ length: 9 }, (_, i) =>
    i === 0 || i === 8 ? rand(0, 0.025) : rand(0.01, 0.12),
  )
  const wander = Array.from({ length: 3 }, () => rand(-0.05, 0.05))
  return { jag, wander }
}

function getPageMetrics(w: number, h: number) {
  return {
    cx: w * 0.2,
    cy: h * 0.47,
    pw: w * 0.24,
    ph: h * 0.58,
    angle: (-7 * Math.PI) / 180,
  }
}

function pageOutlinePoints(w: number, h: number, jag: number[]): Pt[] {
  const { pw, ph } = getPageMetrics(w, h)
  const ox = -pw / 2
  const oy = -ph / 2
  const top: Pt[] = jag.map((j, i) => ({
    x: ox + (i / (jag.length - 1)) * pw,
    y: oy + j * ph,
  }))
  return [
    ...top,
    { x: ox + pw, y: oy + ph },
    { x: ox, y: oy + ph },
    { x: ox, y: top[0].y },
  ]
}

function cornerCurlPoints(w: number, h: number): Pt[] {
  const { pw, ph } = getPageMetrics(w, h)
  const ox = -pw / 2
  const oy = -ph / 2
  return [
    { x: ox + pw * 0.05, y: oy + ph * 0.86 },
    { x: ox + pw * 0.015, y: oy + ph * 0.94 },
    { x: ox + pw * 0.09, y: oy + ph * 0.985 },
    { x: ox + pw * 0.155, y: oy + ph * 0.93 },
    { x: ox + pw * 0.105, y: oy + ph * 0.855 },
  ]
}

function ruledLine1(w: number, h: number): Pt[] {
  const { pw, ph } = getPageMetrics(w, h)
  const ox = -pw / 2
  const oy = -ph / 2
  return [
    { x: ox + pw * 0.16, y: oy + ph * 0.32 },
    { x: ox + pw * 0.84, y: oy + ph * 0.33 },
  ]
}

function ruledLine2(w: number, h: number): Pt[] {
  const { pw, ph } = getPageMetrics(w, h)
  const ox = -pw / 2
  const oy = -ph / 2
  return [
    { x: ox + pw * 0.16, y: oy + ph * 0.48 },
    { x: ox + pw * 0.78, y: oy + ph * 0.485 },
  ]
}

function pageAttachPoint(w: number, h: number): Pt {
  const { cx, cy, pw, ph, angle } = getPageMetrics(w, h)
  const lx = pw * 0.48
  const ly = ph * 0.1
  return {
    x: cx + lx * Math.cos(angle) - ly * Math.sin(angle),
    y: cy + lx * Math.sin(angle) + ly * Math.cos(angle),
  }
}

function xMarkCenter(w: number, h: number): Pt {
  return { x: w * 0.8, y: h * 0.6 }
}

function pathWaypoints(w: number, h: number, wander: number[]): Pt[] {
  const start = pageAttachPoint(w, h)
  const mark = xMarkCenter(w, h)
  return [
    start,
    { x: w * 0.4, y: h * (0.28 + wander[0]) },
    { x: w * 0.56, y: h * (0.74 + wander[1]) },
    { x: w * 0.7, y: h * (0.48 + wander[2]) },
    { x: mark.x - w * 0.05, y: mark.y - h * 0.02 },
  ]
}

// Catmull-Rom sampling so the progressively-revealed tip of the path
// follows the real curve, not a straight line toward the next waypoint.
function catmullRom(pts: Pt[], segments = 10): Pt[] {
  if (pts.length < 3) return pts
  const get = (i: number) => pts[Math.max(0, Math.min(pts.length - 1, i))]
  const out: Pt[] = []
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = get(i - 1)
    const p1 = get(i)
    const p2 = get(i + 1)
    const p3 = get(i + 2)
    for (let s = 0; s < segments; s++) {
      const t = s / segments
      const t2 = t * t
      const t3 = t2 * t
      out.push({
        x:
          0.5 *
          (2 * p1.x +
            (p2.x - p0.x) * t +
            (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
            (3 * p1.x - p0.x - 3 * p2.x + p3.x) * t3),
        y:
          0.5 *
          (2 * p1.y +
            (p2.y - p0.y) * t +
            (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
            (3 * p1.y - p0.y - 3 * p2.y + p3.y) * t3),
      })
    }
  }
  out.push(pts[pts.length - 1])
  return out
}

function xStroke1(w: number, h: number): Pt[] {
  const c = xMarkCenter(w, h)
  const s = Math.min(w, h) * 0.075
  return [
    { x: c.x - s, y: c.y - s },
    { x: c.x + s, y: c.y + s },
  ]
}

function xStroke2(w: number, h: number): Pt[] {
  const c = xMarkCenter(w, h)
  const s = Math.min(w, h) * 0.075
  return [
    { x: c.x - s, y: c.y + s },
    { x: c.x + s, y: c.y - s },
  ]
}

function xCirclePoints(w: number, h: number): Pt[] {
  const c = xMarkCenter(w, h)
  const s = Math.min(w, h) * 0.075
  const r = s * 1.9
  const steps = 22
  const pts: Pt[] = []
  for (let i = 0; i <= steps; i++) {
    // slightly more than a full loop, so the ends overlap like a hand-drawn circle
    const a = ((-15 + (i / steps) * 375) * Math.PI) / 180
    pts.push({ x: c.x + Math.cos(a) * r, y: c.y + Math.sin(a) * r * 0.85 })
  }
  return pts
}

function revealPoints(points: Pt[], fraction: number): Pt[] {
  if (points.length === 0 || fraction <= 0) return []
  if (fraction >= 1) return points
  const segments = points.length - 1
  const exact = segments * fraction
  const idx = Math.floor(exact)
  const t = exact - idx
  const out = points.slice(0, idx + 1)
  if (idx < points.length - 1) {
    const p0 = points[idx]
    const p1 = points[idx + 1]
    out.push({ x: p0.x + (p1.x - p0.x) * t, y: p0.y + (p1.y - p0.y) * t })
  }
  return out
}

function toTuples(points: Pt[]): [number, number][] {
  return points.map((p) => [p.x, p.y])
}

function readColors(): Colors {
  if (typeof document === 'undefined') {
    return {
      text: '#2e2a23',
      muted: '#6f675a',
      terracotta: '#d4552a',
      sage: '#17697a',
      gold: '#c0932f',
    }
  }
  const style = getComputedStyle(document.documentElement)
  const pick = (name: string, fallback: string) =>
    style.getPropertyValue(name).trim() || fallback
  return {
    text: pick('--color-text', '#2e2a23'),
    muted: pick('--color-text-muted', '#6f675a'),
    terracotta: pick('--color-terracotta', '#d4552a'),
    sage: pick('--color-sage', '#17697a'),
    gold: pick('--color-gold', '#c0932f'),
  }
}

function buildStrokes(w: number, h: number, spec: Spec): StrokeDef[] {
  const { cx, cy, angle } = getPageMetrics(w, h)
  const rotate = { cx, cy, angle }
  return [
    {
      points: pageOutlinePoints(w, h, spec.jag),
      colorKey: 'text',
      strokeWidth: 1.75,
      roughness: 1.6,
      bowing: 1,
      seed: 1,
      alpha: 0.9,
      start: 0,
      end: 0.32,
      rotate,
    },
    {
      points: cornerCurlPoints(w, h),
      colorKey: 'muted',
      strokeWidth: 1.4,
      roughness: 2,
      bowing: 1.5,
      seed: 2,
      alpha: 0.75,
      start: 0.28,
      end: 0.42,
      rotate,
    },
    {
      points: ruledLine1(w, h),
      colorKey: 'muted',
      strokeWidth: 1,
      roughness: 1,
      bowing: 0.5,
      seed: 3,
      alpha: 0.4,
      start: 0.36,
      end: 0.44,
      rotate,
    },
    {
      points: ruledLine2(w, h),
      colorKey: 'muted',
      strokeWidth: 1,
      roughness: 1,
      bowing: 0.5,
      seed: 4,
      alpha: 0.4,
      start: 0.42,
      end: 0.5,
      rotate,
    },
    {
      points: catmullRom(pathWaypoints(w, h, spec.wander)),
      colorKey: 'sage',
      strokeWidth: 1.75,
      roughness: 1.5,
      bowing: 1.2,
      seed: 5,
      dash: [8, 7],
      alpha: 0.85,
      start: 0.48,
      end: 0.86,
    },
    {
      points: xStroke1(w, h),
      colorKey: 'terracotta',
      strokeWidth: 2.75,
      roughness: 1.3,
      bowing: 0.8,
      seed: 6,
      alpha: 0.95,
      start: 0.84,
      end: 0.92,
    },
    {
      points: xStroke2(w, h),
      colorKey: 'terracotta',
      strokeWidth: 2.75,
      roughness: 1.3,
      bowing: 0.8,
      seed: 7,
      alpha: 0.95,
      start: 0.9,
      end: 0.98,
    },
    {
      points: xCirclePoints(w, h),
      colorKey: 'terracotta',
      strokeWidth: 1.5,
      roughness: 2.2,
      bowing: 1.5,
      seed: 8,
      alpha: 0.5,
      start: 0.94,
      end: 1,
    },
  ]
}

function renderScene(
  ctx: CanvasRenderingContext2D,
  rc: RoughCanvas,
  w: number,
  h: number,
  spec: Spec,
  progress: number,
) {
  if (w <= 0 || h <= 0) return
  ctx.clearRect(0, 0, w, h)
  const colors = readColors()
  const strokes = buildStrokes(w, h, spec)

  for (const stroke of strokes) {
    const span = stroke.end - stroke.start
    const local =
      span > 0
        ? (progress - stroke.start) / span
        : progress >= stroke.end
          ? 1
          : 0
    const clamped = Math.max(0, Math.min(1, local))
    if (clamped <= 0) continue

    const revealed = revealPoints(stroke.points, clamped)
    if (revealed.length < 2) continue

    ctx.save()
    ctx.globalAlpha = stroke.alpha
    if (stroke.rotate) {
      ctx.translate(stroke.rotate.cx, stroke.rotate.cy)
      ctx.rotate(stroke.rotate.angle)
    }
    rc.linearPath(toTuples(revealed), {
      stroke: colors[stroke.colorKey],
      strokeWidth: stroke.strokeWidth,
      roughness: stroke.roughness,
      bowing: stroke.bowing,
      seed: stroke.seed,
      strokeLineDash: stroke.dash,
    })
    ctx.restore()
  }
}

export function NotFoundScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const specRef = useRef<Spec | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rc = rough.canvas(canvas)
    if (!specRef.current) specRef.current = createSpec()
    const spec = specRef.current
    const progressRef = { current: 0 }
    let rafId: number | undefined
    let startTime: number | null = null

    const sizeCanvas = () => {
      const rect = container.getBoundingClientRect()
      canvas.width = Math.max(1, Math.round(rect.width))
      canvas.height = Math.max(1, Math.round(rect.height))
    }

    const redrawAtCurrentProgress = () => {
      sizeCanvas()
      renderScene(
        ctx,
        rc,
        canvas.width,
        canvas.height,
        spec,
        progressRef.current,
      )
    }

    sizeCanvas()

    const reduceMotion = window.matchMedia(
      '(prefers-reduced-motion: reduce)',
    ).matches

    if (reduceMotion) {
      progressRef.current = 1
      renderScene(ctx, rc, canvas.width, canvas.height, spec, 1)
    } else {
      const tick = (now: number) => {
        if (startTime === null) startTime = now
        const elapsed = now - startTime
        const progress = Math.min(1, elapsed / TOTAL_MS)
        progressRef.current = progress
        renderScene(ctx, rc, canvas.width, canvas.height, spec, progress)
        if (progress < 1) {
          rafId = requestAnimationFrame(tick)
        }
      }
      rafId = requestAnimationFrame(tick)
    }

    const handleThemeChange = () => redrawAtCurrentProgress()
    window.addEventListener(THEME_CHANGE_EVENT, handleThemeChange)

    const resizeObserver = new ResizeObserver(() => {
      redrawAtCurrentProgress()
    })
    resizeObserver.observe(container)

    return () => {
      window.removeEventListener(THEME_CHANGE_EVENT, handleThemeChange)
      resizeObserver.disconnect()
      if (rafId !== undefined) cancelAnimationFrame(rafId)
    }
  }, [])

  return (
    <div ref={containerRef} className={styles.scene}>
      <canvas ref={canvasRef} className={styles.canvas} aria-hidden="true" />
    </div>
  )
}
