'use client'

import { useState, useCallback, useEffect } from 'react'
import { usePathname, useRouter } from 'next/navigation'
import { FileSystem } from '@/lib/filesystem'
import { executeCommand, completeInput, CommandResult } from './commands'

export type OutputLine = {
  type: 'prompt' | 'output' | 'error'
  content: string
}

const HISTORY_KEY = 'terminal-history'
const MAX_HISTORY = 100
const MAX_OUTPUT = 100

function loadHistory(): string[] {
  if (typeof window === 'undefined') return []
  try {
    const stored = localStorage.getItem(HISTORY_KEY)
    return stored ? JSON.parse(stored) : []
  } catch {
    return []
  }
}

function saveHistory(history: string[]) {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-MAX_HISTORY)))
  } catch {
    // ignore storage errors
  }
}

export function useTerminal(fs: FileSystem) {
  const pathname = usePathname()
  const router = useRouter()
  
  // cwd is derived from pathname
  const cwd = pathname || '/'
  
  const [input, setInput] = useState('')
  const [output, setOutput] = useState<OutputLine[]>([])
  const [history, setHistory] = useState<string[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const [tabCount, setTabCount] = useState(0)
  const [lastTabInput, setLastTabInput] = useState('')

  // load history on mount
  useEffect(() => {
    setHistory(loadHistory())
  }, [])

  // format prompt
  const prompt = cwd === '/' ? '~ $ ' : `~${cwd} $ `

  const addOutput = useCallback((lines: OutputLine[]) => {
    setOutput(prev => [...prev, ...lines].slice(-MAX_OUTPUT))
  }, [])

  const execute = useCallback(() => {
    const trimmed = input.trim()
    
    // add prompt line to output
    addOutput([{ type: 'prompt', content: prompt + trimmed }])
    
    if (!trimmed) {
      setInput('')
      return
    }

    const result: CommandResult = executeCommand(trimmed, { fs, cwd })

    // handle clear command
    if (result.output === '__CLEAR__') {
      setOutput([])
      setInput('')
      // add to history
      if (trimmed && (history.length === 0 || history[history.length - 1] !== trimmed)) {
        const newHistory = [...history, trimmed]
        setHistory(newHistory)
        saveHistory(newHistory)
      }
      setHistoryIndex(-1)
      return
    }

    // add output
    if (result.output) {
      addOutput([{
        type: result.isError ? 'error' : 'output',
        content: result.output,
      }])
    }

    // navigate if needed
    if (result.navigate) {
      router.push(result.navigate)
    }

    // add to history (avoid consecutive duplicates)
    if (trimmed && (history.length === 0 || history[history.length - 1] !== trimmed)) {
      const newHistory = [...history, trimmed]
      setHistory(newHistory)
      saveHistory(newHistory)
    }

    setInput('')
    setHistoryIndex(-1)
    setTabCount(0)
  }, [input, prompt, fs, cwd, history, router, addOutput])

  const handleHistoryUp = useCallback(() => {
    if (history.length === 0) return
    
    const newIndex = historyIndex === -1 
      ? history.length - 1 
      : Math.max(0, historyIndex - 1)
    
    setHistoryIndex(newIndex)
    setInput(history[newIndex])
  }, [history, historyIndex])

  const handleHistoryDown = useCallback(() => {
    if (historyIndex === -1) return
    
    const newIndex = historyIndex + 1
    
    if (newIndex >= history.length) {
      setHistoryIndex(-1)
      setInput('')
    } else {
      setHistoryIndex(newIndex)
      setInput(history[newIndex])
    }
  }, [history, historyIndex])

  const handleTab = useCallback(() => {
    // track consecutive tabs on same input
    const isConsecutiveTab = input === lastTabInput
    const newTabCount = isConsecutiveTab ? tabCount + 1 : 1
    
    setTabCount(newTabCount)
    setLastTabInput(input)

    const { completed, options } = completeInput(input, { fs, cwd })

    if (completed !== input) {
      // single completion
      setInput(completed)
      setTabCount(0)
    } else if (options.length > 0 && newTabCount >= 2) {
      // show options on double tab
      addOutput([
        { type: 'prompt', content: prompt + input },
        { type: 'output', content: options.join('  ') },
      ])
    }
  }, [input, lastTabInput, tabCount, fs, cwd, prompt, addOutput])

  return {
    cwd,
    prompt,
    input,
    setInput,
    output,
    execute,
    handleHistoryUp,
    handleHistoryDown,
    handleTab,
  }
}

