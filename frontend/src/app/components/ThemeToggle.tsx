'use client'

import { useSyncExternalStore } from 'react'
import { toggleTheme, THEME_CHANGE_EVENT, type Theme } from '@/lib/theme'

function subscribe(onStoreChange: () => void) {
  window.addEventListener(THEME_CHANGE_EVENT, onStoreChange)
  return () => window.removeEventListener(THEME_CHANGE_EVENT, onStoreChange)
}

function getSnapshot(): Theme {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'
}

function getServerSnapshot(): Theme {
  return 'light'
}

export function ThemeToggle() {
  const theme = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggleTheme}
      aria-label="Toggle color theme"
    >
      {theme === 'dark' ? '☀ day' : '☾ night'}
    </button>
  )
}
