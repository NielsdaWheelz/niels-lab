'use client'

import { useState, useCallback, useRef } from 'react'
import { usePathname, useRouter } from 'next/navigation'
import { FileSystem } from '@/lib/filesystem'
import { executeCommand, completeInput, CommandResult, isSlashCommand } from './commands'

export type OutputLine = {
  type: 'prompt' | 'output' | 'error' | 'thinking' | 'user' | 'assistant'
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

export type ChatHandler = (
  message: string,
  context: { cwd: string; fs: FileSystem }
) => Promise<{ response: string | React.ReactNode; isError?: boolean }>

export function useTerminal(fs: FileSystem, onChat?: ChatHandler) {
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

  // Abort controller for cancelling LLM requests
  const abortControllerRef = useRef<AbortController | null>(null)

  // format prompt - different for commands vs chat
  const prompt = cwd === '/' ? '~ $ ' : `~${cwd} $ `

  const addOutput = useCallback((lines: OutputLine[]) => {
    setOutput(prev => [...prev, ...lines].slice(-MAX_OUTPUT))
  }, [])

  const execute = useCallback(async () => {
    const trimmed = input.trim()

    if (!trimmed) {
      setInput('')
      return
    }

    const isCommand = trimmed.startsWith('/')

    // Add user message/prompt to output
    addOutput([{
      type: isCommand ? 'prompt' : 'user',
      content: isCommand ? (prompt + trimmed) : trimmed,
      id: generateLineId()
    }])

    // Set status to running
    if (isCommand) {
      setStatus(`running ${trimmed.split(/\s+/)[0]}...`)
    }

    // Execute as slash command
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

    // handle chat message (non-slash input)
    if (result.output === '__CHAT__') {
      // add to history
      if (trimmed && (history.length === 0 || history[history.length - 1] !== trimmed)) {
        setHistory(prev => [...prev, trimmed].slice(-MAX_HISTORY))
      }
      setHistoryIndex(-1)
      setInput('')

      if (onChat) {
        // LLM handler provided - use it
        setIsThinking(true)
        setStatus('thinking...')
        addOutput([{ type: 'thinking', content: '', id: generateLineId() }])

        // Create abort controller for this request
        abortControllerRef.current = new AbortController()

        try {
          const { response, isError } = await onChat(trimmed, { cwd, fs })

          // Remove thinking line and add response
          setOutput(prev => {
            const withoutThinking = prev.filter(line => line.type !== 'thinking')
            const newLine: OutputLine = {
              type: isError ? 'error' : 'assistant',
              content: response,
              id: generateLineId(),
              isTyping: typeof response === 'string' && !isError,
            }
            return [...withoutThinking, newLine].slice(-MAX_OUTPUT)
          })

          setStatus(isError ? 'error' : (typeof response === 'string' ? 'typing...' : 'idle'))
        } catch (err) {
          // Handle abort or error
          setOutput(prev => {
            const withoutThinking = prev.filter(line => line.type !== 'thinking')
            if ((err as Error).name === 'AbortError') {
              return withoutThinking // silently remove thinking on abort
            }
            const errorLine: OutputLine = {
              type: 'error',
              content: `error: ${(err as Error).message || 'something went wrong'}`,
              id: generateLineId(),
            }
            return [...withoutThinking, errorLine].slice(-MAX_OUTPUT)
          })
          setStatus('error')
        } finally {
          setIsThinking(false)
          abortControllerRef.current = null
        }
      } else {
        // No LLM handler - show placeholder message
        addOutput([{
          type: 'assistant',
          content: "i'm not connected to an AI yet. use /help to see available commands.",
          id: generateLineId(),
          isTyping: true,
        }])
        setStatus('typing...')
      }

      setTabCount(0)
      return
    }

    // add command output
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
  }, [input, prompt, fs, cwd, history, router, addOutput, onChat])

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
    // Only tab-complete for commands
    if (!input.startsWith('/')) {
      return
    }

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

  // Cancel ongoing LLM request
  const cancelRequest = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    setIsThinking(false)
    setStatus('idle')
  }, [])

  // For manual LLM integration
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
        type: isError ? 'error' : 'assistant',
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
    cancelRequest,
    isSlashCommand: (text: string) => isSlashCommand(text),
  }
}
