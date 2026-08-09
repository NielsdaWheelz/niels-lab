'use client'

import { useSyncExternalStore } from 'react'
import { getTheme, toggleTheme, THEME_CHANGE_EVENT } from '@/lib/theme'

function subscribe(onStoreChange: () => void) {
  window.addEventListener(THEME_CHANGE_EVENT, onStoreChange)
  return () => window.removeEventListener(THEME_CHANGE_EVENT, onStoreChange)
}

export function ThemeToggle() {
  // getTheme decides the current theme everywhere, including the server
  // snapshot: dark is canonical, so SSR HTML offers "day".
  const theme = useSyncExternalStore(subscribe, getTheme, getTheme)

  return (
    <button type="button" className="theme-toggle" onClick={toggleTheme}>
      {theme === 'dark' ? '☀ day' : '☾ night'}
    </button>
  )
}
