'use client'

import { useEffect, useState } from 'react'
import {
  getTheme,
  toggleTheme,
  THEME_CHANGE_EVENT,
  type Theme,
} from '@/lib/theme'

export function ThemeToggle() {
  // The server cannot know the visitor's theme, so the first client render
  // must match SSR exactly ("☀ day", the dark-canonical assumption) and the
  // real label arrives after mount. A useSyncExternalStore snapshot mismatch
  // here makes React 19 client-render the whole root and strip the init
  // script's data-theme off <html> (observed) — do not reintroduce it.
  const [theme, setTheme] = useState<Theme | null>(null)

  useEffect(() => {
    const sync = () => setTheme(getTheme())
    sync()
    window.addEventListener(THEME_CHANGE_EVENT, sync)
    return () => window.removeEventListener(THEME_CHANGE_EVENT, sync)
  }, [])

  return (
    <button type="button" className="theme-toggle" onClick={toggleTheme}>
      {(theme ?? 'dark') === 'dark' ? '☀ day' : '☾ night'}
    </button>
  )
}
