'use client'

import React, { useRef, useEffect, useState, KeyboardEvent } from 'react'
import { FileSystem } from '@/lib/filesystem'
import { useTerminal, OutputLine } from './useTerminal'

interface TerminalProps {
  filesystem: FileSystem
}

function ThinkingIndicator() {
  return (
    <span className="assistant-thinking">
      <span className="thinking-dot" />
      <span className="thinking-dot" />
      <span className="thinking-dot" />
    </span>
  )
}

function OutputLineContent({ line }: { line: OutputLine }) {
  if (line.type === 'thinking') {
    return <ThinkingIndicator />
  }
  
  if (line.type === 'output') {
    return <>{formatOutput(line.content)}</>
  }
  
  return <>{line.content}</>
}

export function Terminal({ filesystem }: TerminalProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const outputRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const [focused, setFocused] = useState(false)
  const [mounted, setMounted] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const {
    prompt,
    input,
    setInput,
    output,
    execute,
    handleHistoryUp,
    handleHistoryDown,
    handleTab,
    isThinking,
  } = useTerminal(filesystem)

  useEffect(() => {
    setMounted(true)
  }, [])

  // auto-scroll output
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [output])

  // global ctrl+k handler
  useEffect(() => {
    const handleGlobalKeyDown = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'k' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        setExpanded(true)
        setTimeout(() => inputRef.current?.focus(), 50)
      }
    }

    window.addEventListener('keydown', handleGlobalKeyDown)
    return () => window.removeEventListener('keydown', handleGlobalKeyDown)
  }, [])

  // click outside to collapse
  useEffect(() => {
    if (!expanded) return

    const handleClickOutside = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        setExpanded(false)
      }
    }

    // delay to avoid immediate collapse
    const timeoutId = setTimeout(() => {
      document.addEventListener('click', handleClickOutside)
    }, 100)

    return () => {
      clearTimeout(timeoutId)
      document.removeEventListener('click', handleClickOutside)
    }
  }, [expanded])

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    switch (e.key) {
      case 'Enter':
        e.preventDefault()
        execute()
        break
      case 'ArrowUp':
        e.preventDefault()
        handleHistoryUp()
        break
      case 'ArrowDown':
        e.preventDefault()
        handleHistoryDown()
        break
      case 'Tab':
        e.preventDefault()
        handleTab()
        break
      case 'Escape':
        e.preventDefault()
        inputRef.current?.blur()
        setExpanded(false)
        break
    }
  }

  const handleToggle = () => {
    const willExpand = !expanded
    setExpanded(willExpand)
    if (willExpand) {
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }

  if (!mounted) {
    return null
  }

  const hasOutput = output.length > 0

  return (
    <div
      ref={panelRef}
      className={`assistant-panel ${expanded ? 'expanded' : ''} ${focused ? 'focused' : ''}`}
    >
      {/* Collapsed state: just the input bar */}
      {!expanded && (
        <div className="assistant-collapsed" onClick={handleToggle}>
          <span className="assistant-prompt-hint">~</span>
          <span className="assistant-placeholder">use help to get started</span>
          <kbd className="assistant-shortcut">⌘K</kbd>
        </div>
      )}

      {/* Expanded state: output + input */}
      {expanded && (
        <>
          {/* Output area */}
          {hasOutput && (
            <div ref={outputRef} className="assistant-output">
              {output.map((line, i) => (
                <div
                  key={i}
                  className={`assistant-line assistant-line--${line.type}`}
                >
                  <OutputLineContent line={line} />
                </div>
              ))}
            </div>
          )}

          {/* Input area */}
          <div className="assistant-input-row">
            <span className="assistant-prompt">{prompt}</span>
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              className="assistant-input"
              placeholder={hasOutput ? '' : 'type a command or use help to get started...'}
              spellCheck={false}
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              disabled={isThinking}
              aria-label="Assistant input"
            />
            {!isThinking && input === '' && (
              <span className="assistant-escape-hint">esc to close</span>
            )}
          </div>
        </>
      )}
    </div>
  )
}

// Format output with colors for directories
function formatOutput(content: string): React.ReactNode {
  // Check if this looks like ls output (space-separated items with slashes)
  if (content.includes('/') && !content.includes('\n') && content.split(/\s+/).length <= 20) {
    const parts = content.split(/\s+/).filter(Boolean)
    return (
      <>
        {parts.map((part, i) => (
          <span key={i}>
            {i > 0 && '  '}
            <span className={part.endsWith('/') ? 'assistant-dir' : ''}>
              {part}
            </span>
          </span>
        ))}
      </>
    )
  }
  return <>{content}</>
}
