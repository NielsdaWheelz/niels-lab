'use client'

import { useEffect, useRef, useState } from 'react'
import rough from 'roughjs'

interface NeuralPathwayProps {
  containerRef: React.RefObject<HTMLElement>
  nodeSelectors?: string[]
  className?: string
  hoveredElement?: HTMLElement | null
}

interface Node {
  element: HTMLElement
  x: number
  y: number
  id: string
  isHovered?: boolean
}

export function NeuralPathway({ 
  containerRef, 
  nodeSelectors = ['a'],
  className = '',
  hoveredElement = null
}: NeuralPathwayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationFrameRef = useRef<number>()
  const nodesRef = useRef<Node[]>([])
  const hoverStateRef = useRef<HTMLElement | null>(null)

  // Sync hover state
  useEffect(() => {
    hoverStateRef.current = hoveredElement || null
  }, [hoveredElement])

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rc = rough.canvas(canvas)

    const updateNodes = () => {
      nodesRef.current = []
      const elements = container.querySelectorAll(nodeSelectors.join(', '))
      const hovered = hoverStateRef.current
      
      elements.forEach((el, index) => {
        if (el instanceof HTMLElement) {
          const rect = el.getBoundingClientRect()
          const containerRect = container.getBoundingClientRect()
          nodesRef.current.push({
            element: el,
            x: rect.left + rect.width / 2 - containerRect.left,
            y: rect.top + rect.height / 2 - containerRect.top,
            id: `node-${index}`,
            isHovered: el === hovered,
          })
        }
      })
    }

    const resize = () => {
      if (!container) return
      const rect = container.getBoundingClientRect()
      canvas.width = rect.width
      canvas.height = rect.height
      updateNodes()
    }

    const draw = () => {
      if (!ctx) return
      
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const nodes = nodesRef.current
      const hasHovered = nodes.some(n => n.isHovered)
      const hoveredNode = nodes.find(n => n.isHovered)

      if (nodes.length < 2) {
        animationFrameRef.current = requestAnimationFrame(draw)
        return
      }

      // Draw connections between nearby nodes
      ctx.save()
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const n1 = nodes[i]
          const n2 = nodes[j]
          
          const dist = Math.hypot(n2.x - n1.x, n2.y - n1.y)
          
          // Increase connection distance and opacity on hover
          const MAX_DISTANCE = hasHovered ? 250 : 150
          const MIN_DISTANCE = 25
          
          // Prefer horizontal connections (for nav) but allow vertical ones too
          const horizontalDist = Math.abs(n2.x - n1.x)
          const verticalDist = Math.abs(n2.y - n1.y)
          
          if (dist < MAX_DISTANCE && dist > MIN_DISTANCE) {
            // Slightly favor connections where horizontal distance isn't too large
            // (avoids connecting wrapped nav items)
            if (!hasHovered && horizontalDist > 200 && verticalDist < 40) continue
            
            // Check if this connection involves the hovered node
            const involvesHovered = n1.isHovered || n2.isHovered
            const isDirectToHovered = involvesHovered && (n1.isHovered || n2.isHovered)
            
            // Base opacity scales with hover state
            let baseOpacity = hasHovered ? 0.25 : 0.1
            if (isDirectToHovered) {
              baseOpacity = 0.4 // Stronger for connections to hovered node
            }
            
            const connectionOpacity = (1 - dist / MAX_DISTANCE) * baseOpacity
            ctx.globalAlpha = connectionOpacity
            
            // Make hovered connections slightly thicker
            const strokeWidth = isDirectToHovered ? 0.8 : 0.5
            
            rc.line(n1.x, n1.y, n2.x, n2.y, {
              stroke: 'var(--color-text)',
              strokeWidth: strokeWidth,
              roughness: 1.5,
              bowing: 0.8,
            })
          }
        }

        // Draw node circles - more prominent when hovered
        const node = nodes[i]
        const nodeOpacity = node.isHovered ? 0.15 : 0.06
        const nodeSize = node.isHovered ? 4 : 2.5
        
        ctx.globalAlpha = nodeOpacity
        rc.circle(node.x, node.y, nodeSize, {
          stroke: 'var(--color-text)',
          strokeWidth: node.isHovered ? 0.8 : 0.5,
          roughness: 1,
          fill: 'var(--color-text)',
          fillStyle: 'solid',
        })
      }
      
      // On hover, draw additional secondary connections (more complex network)
      if (hasHovered && hoveredNode) {
        // Connect hovered node to nodes it might not directly connect to
        for (let i = 0; i < nodes.length; i++) {
          const node = nodes[i]
          if (node === hoveredNode) continue
          
          const dist = Math.hypot(node.x - hoveredNode.x, node.y - hoveredNode.y)
          
          // Secondary connections to nodes further away
          if (dist > 150 && dist < 350) {
            const connectionOpacity = (1 - (dist - 150) / 200) * 0.12
            ctx.globalAlpha = connectionOpacity
            
            // Slightly wavier connection for secondary paths
            rc.line(hoveredNode.x, hoveredNode.y, node.x, node.y, {
              stroke: 'var(--color-text)',
              strokeWidth: 0.4,
              roughness: 2,
              bowing: 1.2,
            })
          }
        }
      }
      
      ctx.restore()

      animationFrameRef.current = requestAnimationFrame(draw)
    }

    resize()
    window.addEventListener('resize', resize)
    
    // Update nodes when container changes (e.g., on route change)
    const observer = new MutationObserver(() => {
      updateNodes()
    })
    observer.observe(container, { childList: true, subtree: true })

    // Update nodes on hover changes
    const interval = setInterval(() => {
      updateNodes()
    }, 50) // Check hover state frequently

    animationFrameRef.current = requestAnimationFrame(draw)

    return () => {
      window.removeEventListener('resize', resize)
      observer.disconnect()
      clearInterval(interval)
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [containerRef, nodeSelectors, hoveredElement])

  return (
    <canvas
      ref={canvasRef}
      className={`neural-pathway ${className}`}
      aria-hidden="true"
    />
  )
}

