'use client'

import { useState } from 'react'
import { TextReveal } from './TextReveal'

const bio1 = "i like old movies and older books. i play an unhealthy amount of soccer ever since my doctor forbade me from playing an unhealthy amount of hockey. currently attending fractal bootcamp."
const bio2 = "interested in neural networks, systems, and building things that work."

export function HomeContent() {
  const [showSecond, setShowSecond] = useState(false)

  return (
    <>
      <p>
        <TextReveal 
          text={bio1} 
          speed={15} 
          onComplete={() => setShowSecond(true)}
        />
      </p>
      {showSecond && (
        <p>
          <TextReveal text={bio2} speed={15} delay={200} />
        </p>
      )}
    </>
  )
}

