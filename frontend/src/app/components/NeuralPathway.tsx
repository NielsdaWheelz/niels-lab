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

      // If only one node (heading) and it's hovered, create wavy web around it
      if (nodes.length === 1 && hasHovered && hoveredNode) {
        const centerX = hoveredNode.x
        const centerY = hoveredNode.y
        const element = hoveredNode.element
        const rect = element.getBoundingClientRect()
        const containerRect = container.getBoundingClientRect()
        
        // Get text bounds to create web around the actual text area
        const textLeft = rect.left - containerRect.left
        const textRight = rect.right - containerRect.left
        const textTop = rect.top - containerRect.top
        const textBottom = rect.bottom - containerRect.top
        const textWidth = textRight - textLeft
        const textHeight = textBottom - textTop
        
        // Create organic connection points around the text
        const connectionPoints: Array<{ x: number; y: number }> = []
        const numPoints = 15
        
        // Points around the text perimeter and slightly beyond
        const padding = 30
        
        for (let i = 0; i < numPoints; i++) {
          const progress = i / numPoints
          let x, y
          
          if (progress < 0.25) {
            // Top edge
            const t = progress * 4
            x = textLeft + textWidth * t - padding + (Math.random() - 0.5) * 20
            y = textTop - padding + (Math.random() - 0.5) * 15
          } else if (progress < 0.5) {
            // Right edge
            const t = (progress - 0.25) * 4
            x = textRight + padding + (Math.random() - 0.5) * 15
            y = textTop + textHeight * t + (Math.random() - 0.5) * 20
          } else if (progress < 0.75) {
            // Bottom edge
            const t = (progress - 0.5) * 4
            x = textRight - textWidth * t + padding + (Math.random() - 0.5) * 20
            y = textBottom + padding + (Math.random() - 0.5) * 15
          } else {
            // Left edge
            const t = (progress - 0.75) * 4
            x = textLeft - padding + (Math.random() - 0.5) * 15
            y = textBottom - textHeight * t + (Math.random() - 0.5) * 20
          }
          
          connectionPoints.push({ x, y })
        }
        
        // Add some interior points for denser web
        for (let i = 0; i < 8; i++) {
          connectionPoints.push({
            x: textLeft + Math.random() * textWidth,
            y: textTop + Math.random() * textHeight,
          })
        }
        
        ctx.save()
        
        // Draw wavy web of connections
        for (let i = 0; i < connectionPoints.length; i++) {
          for (let j = i + 1; j < connectionPoints.length; j++) {
            const p1 = connectionPoints[i]
            const p2 = connectionPoints[j]
            const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y)
            
            // Connect points that are reasonably close (web-like pattern)
            if (dist > 20 && dist < 120) {
              const connectionOpacity = (1 - dist / 120) * 0.25
              ctx.globalAlpha = connectionOpacity
              
              // Wavy, organic lines with higher roughness for hand-drawn feel
              rc.line(p1.x, p1.y, p2.x, p2.y, {
                stroke: 'var(--color-text)',
                strokeWidth: 0.5,
                roughness: 2.5,
                bowing: 1.8,
              })
            }
          }
          
          // Connect points to center (text) for more web-like effect
          const p = connectionPoints[i]
          const distToCenter = Math.hypot(p.x - centerX, p.y - centerY)
          if (distToCenter > 30 && distToCenter < 100) {
            const connectionOpacity = (1 - distToCenter / 100) * 0.2
            ctx.globalAlpha = connectionOpacity
            
            rc.line(p.x, p.y, centerX, centerY, {
              stroke: 'var(--color-text)',
              strokeWidth: 0.5,
              roughness: 2.2,
              bowing: 1.5,
            })
          }
        }
        
        ctx.restore()
        animationFrameRef.current = requestAnimationFrame(draw)
        return
      }

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

