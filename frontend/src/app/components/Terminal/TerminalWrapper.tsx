'use client'

import { useMemo } from 'react'
import { buildFileSystem } from '@/lib/filesystem'
import { handleChat } from '@/lib/nielsGpt'
import { Terminal } from './index'

export type ContentData = {
  blogPosts: Array<{ slug: string; metadata: Record<string, string>; content: string }>
  projects: Array<{ slug: string; metadata: Record<string, string>; content: string }>
  braindumps: Array<{ slug: string; metadata: Record<string, string>; content: string }>
  cv: { slug?: string; metadata: Record<string, string>; content: string }
}

interface TerminalWrapperProps {
  data: ContentData
}

export function TerminalWrapper({ data }: TerminalWrapperProps) {
  const filesystem = useMemo(() =>
    buildFileSystem(data.blogPosts, data.projects, data.braindumps, data.cv),
    [data]
  )

  return <Terminal filesystem={filesystem} onChat={handleChat} />
}
