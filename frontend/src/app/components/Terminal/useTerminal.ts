'use client'

import { useState, useCallback } from 'react'
import { usePathname, useRouter } from 'next/navigation'
import { FileSystem } from '@/lib/filesystem'
import { executeCommand, completeInput, CommandResult } from './commands'

export type OutputLine = {
  type: 'prompt' | 'output' | 'error' | 'thinking'
  content: string | React.ReactNode
  id: string // unique id for tracking typewriter state
  isTyping?: boolean // whether this line is currently typing
}

export type TerminalStatus = 'idle' | 'typing' | 'thinking' | 'error' | string

const MAX_OUTPUT = 50
const MAX_HISTORY = 50

// Simple ID generator
let lineIdCounter = 0
function generateLineId(): string {
  return `line-${++lineIdCounter}-${Date.now()}`
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
  const [isThinking, setIsThinking] = useState(false)
  const [status, setStatus] = useState<TerminalStatus>('idle')

  // format prompt
  const prompt = cwd === '/' ? '~ $ ' : `~${cwd} $ `

  const addOutput = useCallback((lines: OutputLine[]) => {
    setOutput(prev => [...prev, ...lines].slice(-MAX_OUTPUT))
  }, [])

  const execute = useCallback(() => {
    const trimmed = input.trim()

    // add prompt line to output
    addOutput([{ type: 'prompt', content: prompt + trimmed, id: generateLineId() }])

    if (!trimmed) {
      setInput('')
      return
    }

    // Set status to running
    setStatus(`running ${trimmed.split(/\s+/)[0]}...`)

    // For future LLM integration: detect if this is a natural language query
    // For now, treat everything as a command
    const result: CommandResult = executeCommand(trimmed, { fs, cwd })

    // handle clear command
    if (result.output === '__CLEAR__') {
      setOutput([])
      setInput('')
      setStatus('idle')
      // add to history
      if (trimmed && (history.length === 0 || history[history.length - 1] !== trimmed)) {
        setHistory(prev => [...prev, trimmed].slice(-MAX_HISTORY))
      }
      setHistoryIndex(-1)
      return
    }

    // add output
    if (result.output) {
      const lineId = generateLineId()
      const isStringOutput = typeof result.output === 'string'
      const isReactOutput = !isStringOutput && typeof result.output === 'object' && result.output !== null

      addOutput([{
        type: result.isError ? 'error' : 'output',
        content: result.output,
        id: lineId,
        // Only string outputs get typewriter effect (not errors)
        isTyping: isStringOutput && !result.isError,
      }])

      // Set status based on result
      if (result.status) {
        setStatus(result.status)
      } else if (result.isError) {
        setStatus('error')
      } else if (isStringOutput && !result.isError) {
        setStatus('typing...')
      } else if (isReactOutput) {
        // React outputs render immediately
        setStatus('idle')
      } else {
        setStatus('idle')
      }
    } else {
      setStatus('idle')
    }

    // navigate if needed
    if (result.navigate) {
      router.push(result.navigate)
    }

    // add to history (avoid consecutive duplicates)
    if (trimmed && (history.length === 0 || history[history.length - 1] !== trimmed)) {
      setHistory(prev => [...prev, trimmed].slice(-MAX_HISTORY))
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
        { type: 'prompt', content: prompt + input, id: generateLineId() },
        { type: 'output', content: options.join('  '), id: generateLineId() },
      ])
    }
  }, [input, lastTabInput, tabCount, fs, cwd, prompt, addOutput])

  // Called when typewriter finishes
  const onTypewriterComplete = useCallback((lineId: string) => {
    setOutput(prev =>
      prev.map(line =>
        line.id === lineId ? { ...line, isTyping: false } : line
      )
    )
    setStatus('idle')
  }, [])

  // For future LLM integration
  const startThinking = useCallback(() => {
    setIsThinking(true)
    setStatus('thinking...')
    addOutput([{ type: 'thinking', content: '', id: generateLineId() }])
  }, [addOutput])

  const stopThinking = useCallback((response: string | React.ReactNode, isError = false) => {
    setIsThinking(false)
    // Remove the thinking line and add the response
    setOutput(prev => {
      const withoutThinking = prev.filter(line => line.type !== 'thinking')
      const newLine: OutputLine = {
        type: isError ? 'error' : 'output',
        content: response,
        id: generateLineId(),
        isTyping: typeof response === 'string' && !isError,
      }
      return [...withoutThinking, newLine].slice(-MAX_OUTPUT)
    })
    setStatus(isError ? 'error' : (typeof response === 'string' ? 'typing...' : 'idle'))
  }, [])

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
    isThinking,
    startThinking,
    stopThinking,
    status,
    setStatus,
    onTypewriterComplete,
  }
}
