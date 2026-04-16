'use client'

import { TextReveal } from './TextReveal'

interface PageTitleProps {
  children: string
  speed?: number
}

export function PageTitle({ children, speed = 25 }: PageTitleProps) {
  return (
    <h1>
      <TextReveal text={children} speed={speed} showCursor={true} />
    </h1>
  )
}
