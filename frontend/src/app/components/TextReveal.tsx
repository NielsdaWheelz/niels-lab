'use client'

import { useState, useEffect, useRef } from 'react'

interface TextRevealProps {
  text: string
  speed?: number // ms per character
  delay?: number // ms before starting
  className?: string
  showCursor?: boolean
  onComplete?: () => void
}

export function TextReveal({
  text,
  speed = 30,
  delay = 0,
  className = '',
  showCursor = true,
  onComplete,
}: TextRevealProps) {
  const [currentIndex, setCurrentIndex] = useState(0)
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)
  const onCompleteRef = useRef(onComplete)
  const hasCalledComplete = useRef(false)

  // Keep onComplete ref up to date
  useEffect(() => {
    onCompleteRef.current = onComplete
  }, [onComplete])

  // Reset when text changes
  useEffect(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }

    const frameId = requestAnimationFrame(() => {
      hasCalledComplete.current = false
      setCurrentIndex(0)
    })

    return () => cancelAnimationFrame(frameId)
  }, [text])

  // Typing effect
  useEffect(() => {
    if (currentIndex >= text.length) {
      if (!hasCalledComplete.current) {
        hasCalledComplete.current = true
        onCompleteRef.current?.()
      }
      return
    }

    // Calculate when to start (accounting for delay on first character)
    const startTime = currentIndex === 0 ? delay : 0

    timeoutRef.current = setTimeout(() => {
      setCurrentIndex((prev) => prev + 1)
    }, startTime + speed)

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
    }
  }, [currentIndex, text.length, speed, delay])

  const displayedText = text.slice(0, currentIndex)
  const isComplete = currentIndex >= text.length

  return (
    <span className={className}>
      {displayedText}
      {showCursor && !isComplete && <span className="cursor" />}
    </span>
  )
}

// variant for paragraphs that reveals all at once after a delay
export function TextFadeIn({
  children,
  delay = 0,
  className = '',
}: {
  children: React.ReactNode
  delay?: number
  className?: string
}) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const timeout = setTimeout(() => setVisible(true), delay)
    return () => clearTimeout(timeout)
  }, [delay])

  return (
    <div
      className={className}
      style={{
        opacity: visible ? 1 : 0,
        transition: 'opacity 0.3s ease',
      }}
    >
      {children}
    </div>
  )
}
