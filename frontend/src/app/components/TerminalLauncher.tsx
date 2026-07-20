'use client'

import { TERMINAL_OPEN_EVENT } from '@/lib/terminal'

export function TerminalLauncher() {
  return (
    <button
      type="button"
      className="hero-action hero-action-secondary"
      onClick={() => window.dispatchEvent(new Event(TERMINAL_OPEN_EVENT))}
    >
      ask the site
      <kbd>⌘K</kbd>
    </button>
  )
}
