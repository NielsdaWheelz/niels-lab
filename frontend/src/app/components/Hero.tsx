'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { TextReveal } from './TextReveal'
import { NeuralPathway } from './NeuralPathway'

const greeting = "hey, i'm niels"
const bio1 = "i like old movies and older books. i play an unhealthy amount of soccer ever since my doctor forbade me from playing an unhealthy amount of hockey. currently attending fractal bootcamp."
const bio2 = "interested in neural networks, systems, and building things that work."

export function Hero() {
  const [phase, setPhase] = useState(0)
  const [mounted, setMounted] = useState(false)
  const heroRef = useRef<HTMLElement>(null)

  useEffect(() => {
    setMounted(true)
  }, [])

  const handleGreetingComplete = useCallback(() => {
    setTimeout(() => setPhase(1), 400)
  }, [])

  const handleBio1Complete = useCallback(() => {
    setTimeout(() => setPhase(2), 300)
  }, [])

  return (
    <section ref={heroRef} className="hero">
      {/* decorative scribble */}
      <svg 
        className="hero-scribble" 
        viewBox="0 0 120 40" 
        fill="none"
        aria-hidden="true"
      >
        <path 
          d="M5 20 Q 15 5, 30 18 T 55 15 T 80 22 T 105 12 T 115 25"
          stroke="var(--color-sage)"
          strokeWidth="2"
          strokeLinecap="round"
          fill="none"
          className="scribble-path"
        />
      </svg>

      <div className="hero-content">
        <h1 className="hero-greeting">
          {mounted && (
            <TextReveal 
              text={greeting} 
              speed={25} 
              showCursor={true}
              onComplete={handleGreetingComplete}
            />
          )}
        </h1>

        <div className="hero-bio">
          {phase >= 1 && (
            <p>
              <TextReveal 
                text={bio1} 
                speed={25} 
                showCursor={true}
                onComplete={handleBio1Complete}
              />
            </p>
          )}
          {phase >= 2 && (
            <p className="hero-interests">
              <TextReveal 
                text={bio2} 
                speed={25} 
                showCursor={true}
              />
            </p>
          )}
        </div>
      </div>

      {/* small decorative dots */}
      <div className="hero-dots" aria-hidden="true">
        <span className="dot dot-1" />
        <span className="dot dot-2" />
        <span className="dot dot-3" />
      </div>

      {/* neural pathway connections */}
      <NeuralPathway 
        containerRef={heroRef} 
        nodeSelectors={['h1', 'p', '.dot']}
      />
    </section>
  )
}

