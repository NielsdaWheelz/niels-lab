export type Theme = 'light' | 'dark'

const THEME_STORAGE_KEY = 'theme'
export const THEME_CHANGE_EVENT = 'themechange'

export function getTheme(): Theme {
  if (typeof document === 'undefined') return 'light'
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'
}

function setTheme(theme: Theme) {
  if (typeof document === 'undefined') return
  document.documentElement.dataset.theme = theme
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme)
  } catch {
    // private browsing or storage disabled; theme still applies for this page
  }
  window.dispatchEvent(
    new CustomEvent<{ theme: Theme }>(THEME_CHANGE_EVENT, {
      detail: { theme },
    }),
  )
}

export function toggleTheme(): Theme {
  const next: Theme = getTheme() === 'dark' ? 'light' : 'dark'
  setTheme(next)
  return next
}

/**
 * Runs before paint via an inline script in the document head, so the
 * stored/preferred theme applies without a flash of the wrong palette.
 */
export const themeInitScript = `(function(){try{var t=localStorage.getItem('${THEME_STORAGE_KEY}');if(t!=='dark'&&t!=='light'){t=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'}document.documentElement.dataset.theme=t}catch(e){document.documentElement.dataset.theme='light'}})()`
