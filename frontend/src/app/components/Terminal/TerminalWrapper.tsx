'use client'

import dynamic from 'next/dynamic'
import { useEffect, useMemo, useState } from 'react'
import {
  buildFileSystem,
  type ContentEntry,
  type CVEntry,
} from '@/lib/filesystem'
import { handleChat } from '@/lib/nielsGpt'
import { TERMINAL_OPEN_EVENT } from '@/lib/terminal'

const Terminal = dynamic(
  () => import('./index').then((module) => module.Terminal),
  {
    ssr: false,
    loading: () => (
      <div
        className="assistant-launcher assistant-launcher-loading"
        role="status"
      >
        opening terminal…
      </div>
    ),
  },
)

export type ContentData = {
  projects: ContentEntry[]
  writingPosts?: ContentEntry[]
  cv: CVEntry
}

interface TerminalWrapperProps {
  data: ContentData
}

export function TerminalWrapper({ data }: TerminalWrapperProps) {
  const [activated, setActivated] = useState(false)

  const filesystem = useMemo(
    () =>
      buildFileSystem({
        projects: data.projects,
        writing: data.writingPosts ?? [],
        cv: data.cv,
      }),
    [data.cv, data.projects, data.writingPosts],
  )

  useEffect(() => {
    if (activated) return

    const activate = () => setActivated(true)
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'k' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault()
        activate()
      }
    }

    window.addEventListener(TERMINAL_OPEN_EVENT, activate)
    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener(TERMINAL_OPEN_EVENT, activate)
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [activated])

  if (!activated) {
    return (
      <button
        type="button"
        className="assistant-launcher"
        onClick={() => setActivated(true)}
        aria-label="Open the interactive site terminal"
      >
        <span aria-hidden="true">~</span>
        <span>ask the site</span>
        <kbd>⌘K</kbd>
      </button>
    )
  }

  return (
    <Terminal filesystem={filesystem} onChat={handleChat} initiallyExpanded />
  )
}
