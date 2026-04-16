'use client'

import { useState, useEffect } from 'react'
import { NeuralLoading } from './NeuralLoading'

interface ContentRevealProps {
  children: React.ReactNode
  loadingText?: string
  loadingDuration?: number // ms to show loading state
}

export function ContentReveal({
  children,
  loadingText = 'generating',
  loadingDuration = 800,
}: ContentRevealProps) {
  const [isLoading, setIsLoading] = useState(true)
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const loadingTimer = setTimeout(() => {
      setIsLoading(false)
      // small delay before fade in
      setTimeout(() => setIsVisible(true), 50)
    }, loadingDuration)

    return () => clearTimeout(loadingTimer)
  }, [loadingDuration])

  if (isLoading) {
    return (
      <div style={{ marginTop: '1rem' }}>
        <NeuralLoading text={loadingText} speed={40} />
      </div>
    )
  }

  return (
    <div
      style={{
        opacity: isVisible ? 1 : 0,
        transition: 'opacity 0.4s ease-in-out',
      }}
    >
      {children}
    </div>
  )
}
