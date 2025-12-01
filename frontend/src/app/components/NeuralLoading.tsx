'use client'

import { useState, useEffect } from 'react'

interface NeuralLoadingProps {
  text?: string
  speed?: number
}

export function NeuralLoading({ text = 'loading', speed = 50 }: NeuralLoadingProps) {
  const [displayedText, setDisplayedText] = useState('')
  const [charIndex, setCharIndex] = useState(0)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted) return

    if (charIndex < (text?.length ?? 0)) {
      const timeout = setTimeout(() => {
        setDisplayedText(prev => prev + text[charIndex])
        setCharIndex(prev => prev + 1)
      }, speed + Math.random() * speed) // slight randomness like real generation

      return () => clearTimeout(timeout)
    } else {
      // pause then restart
      const resetTimeout = setTimeout(() => {
        setDisplayedText('')
        setCharIndex(0)
      }, 1000)

      return () => clearTimeout(resetTimeout)
    }
  }, [charIndex, text, speed, mounted])

  return (
    <span style={{ color: '#666' }}>
      {displayedText}
      <span className="cursor" />
    </span>
  )
}

// simpler dots version
export function LoadingDots() {
  return (
    <span style={{ color: '#666' }}>
      <span className="loading-dot">.</span>
      <span className="loading-dot">.</span>
      <span className="loading-dot">.</span>
    </span>
  )
}

