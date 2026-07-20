'use client'

import { useEffect, useRef, useState } from 'react'
import styles from './colophon.module.css'

interface ColorSwatchProps {
  varName: string
  role: string
  light: string
  dark: string
}

// A palette swatch that is also a tiny utility: click it to copy the
// actual `var(--color-…)` reference onto the clipboard. The chip itself
// is painted with that same custom property, so toggling the site theme
// repaints every swatch on the page without a single line of JS here.
export function ColorSwatch({ varName, role, light, dark }: ColorSwatchProps) {
  const [copied, setCopied] = useState(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(`var(${varName})`)
    } catch {
      return
    }
    setCopied(true)
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }
    timeoutRef.current = setTimeout(() => setCopied(false), 1600)
  }

  return (
    <button
      type="button"
      className={styles.swatch}
      onClick={handleCopy}
      aria-label={`Copy ${varName} — ${role}`}
    >
      <span
        className={styles.swatchChip}
        style={{ background: `var(${varName})` }}
        aria-hidden="true"
      />
      <span className={styles.swatchInfo}>
        <code className={styles.swatchToken}>{varName}</code>
        <span className={styles.swatchRole}>{role}</span>
        <span className={styles.swatchHex}>
          light {light} · dark {dark}
        </span>
      </span>
      <span className={styles.swatchCopied} aria-live="polite">
        {copied ? 'copied ✓' : ''}
      </span>
    </button>
  )
}
