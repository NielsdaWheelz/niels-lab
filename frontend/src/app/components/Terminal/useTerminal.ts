'use client'

import { useState, useCallback } from 'react'
import { usePathname, useRouter } from 'next/navigation'
import { FileSystem } from '@/lib/filesystem'
import { executeCommand, completeInput, CommandResult } from './commands'

export type OutputLine = {
  type: 'prompt' | 'output' | 'error' | 'thinking'
  content: string
}

const MAX_OUTPUT = 50
const MAX_HISTORY = 50

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

    // For future LLM integration: detect if this is a natural language query
    // For now, treat everything as a command
    const result: CommandResult = executeCommand(trimmed, { fs, cwd })

    // handle clear command
    if (result.output === '__CLEAR__') {
      setOutput([])
      setInput('')
      // add to history
      if (trimmed && (history.length === 0 || history[history.length - 1] !== trimmed)) {
        setHistory(prev => [...prev, trimmed].slice(-MAX_HISTORY))
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
        { type: 'prompt', content: prompt + input },
        { type: 'output', content: options.join('  ') },
      ])
    }
  }, [input, lastTabInput, tabCount, fs, cwd, prompt, addOutput])

  // For future LLM integration
  const startThinking = useCallback(() => {
    setIsThinking(true)
    addOutput([{ type: 'thinking', content: '' }])
  }, [addOutput])

  const stopThinking = useCallback((response: string, isError = false) => {
    setIsThinking(false)
    // Remove the thinking line and add the response
    setOutput(prev => {
      const withoutThinking = prev.filter(line => line.type !== 'thinking')
      const newLine: OutputLine = { 
        type: isError ? 'error' : 'output', 
        content: response 
      }
      return [...withoutThinking, newLine].slice(-MAX_OUTPUT)
    })
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
  }
}
