'use client'

import { useState, useEffect } from 'react'

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
  const [displayedText, setDisplayedText] = useState('')
  const [isComplete, setIsComplete] = useState(false)
  const [hasStarted, setHasStarted] = useState(false)

  useEffect(() => {
    const startTimeout = setTimeout(() => {
      setHasStarted(true)
    }, delay)

    return () => clearTimeout(startTimeout)
  }, [delay])

  useEffect(() => {
    if (!hasStarted) return

    if (displayedText.length < text.length) {
      const timeout = setTimeout(() => {
        setDisplayedText(text.slice(0, displayedText.length + 1))
      }, speed)

      return () => clearTimeout(timeout)
    } else {
      setIsComplete(true)
      onComplete?.()
    }
  }, [displayedText, text, speed, hasStarted, onComplete])

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

