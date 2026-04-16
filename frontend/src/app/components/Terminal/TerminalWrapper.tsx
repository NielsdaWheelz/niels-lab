'use client'

import { useMemo } from 'react'
import { buildFileSystem, type ContentEntry, type CVEntry } from '@/lib/filesystem'
import { handleChat } from '@/lib/nielsGpt'
import { Terminal } from './index'

export type ContentData = {
  projects: ContentEntry[]
  writingPosts?: ContentEntry[]
  cv: CVEntry
}

interface TerminalWrapperProps {
  data: ContentData
}

export function TerminalWrapper({ data }: TerminalWrapperProps) {
  const filesystem = useMemo(
    () =>
      buildFileSystem({
        projects: data.projects,
        writing: data.writingPosts ?? [],
        cv: data.cv,
      }),
    [data.cv, data.projects, data.writingPosts]
  )

  return <Terminal filesystem={filesystem} onChat={handleChat} />
}
