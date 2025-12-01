'use client'

import { useMemo } from 'react'
import { buildFileSystem } from '@/lib/filesystem'
import { Terminal } from './index'

export type ContentData = {
  blogPosts: Array<{ slug: string; metadata: Record<string, string>; content: string }>
  projects: Array<{ slug: string; metadata: Record<string, string>; content: string }>
  braindumps: Array<{ slug: string; metadata: Record<string, string>; content: string }>
}

interface TerminalWrapperProps {
  data: ContentData
}

export function TerminalWrapper({ data }: TerminalWrapperProps) {
  const filesystem = useMemo(() => 
    buildFileSystem(data.blogPosts, data.projects, data.braindumps),
    [data]
  )

  return <Terminal filesystem={filesystem} />
}

