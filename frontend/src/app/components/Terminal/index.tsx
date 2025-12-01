'use client'

import { useRef, useEffect, useState, KeyboardEvent } from 'react'
import { FileSystem } from '@/lib/filesystem'
import { useTerminal } from './useTerminal'

interface TerminalProps {
  filesystem: FileSystem
}

export function Terminal({ filesystem }: TerminalProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const outputRef = useRef<HTMLDivElement>(null)
  const [focused, setFocused] = useState(false)
  const [mounted, setMounted] = useState(false)

  const {
    prompt,
    input,
    setInput,
    output,
    execute,
    handleHistoryUp,
    handleHistoryDown,
    handleTab,
  } = useTerminal(filesystem)

  // Prevent hydration mismatch by deferring render until client is ready
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
        inputRef.current?.focus()
      }
    }

    window.addEventListener('keydown', handleGlobalKeyDown)
    return () => window.removeEventListener('keydown', handleGlobalKeyDown)
  }, [])

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
        break
    }
  }

  if (!mounted) {
    // Return empty shell during SSR
    return (
      <div
        style={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          height: '140px',
          background: '#0a0a0a',
          borderTop: '1px solid #222',
          fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
          fontSize: '13px',
          zIndex: 50,
          display: 'flex',
          flexDirection: 'column',
        }}
        className="terminal-container"
      >
        <div style={{ flex: 1, overflow: 'auto', padding: '8px 12px' }} />
        <div style={{ display: 'flex', alignItems: 'center', padding: '8px 12px', borderTop: '1px solid #1a1a1a' }} />
      </div>
    )
  }

  return (
    <div
      style={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        height: '140px',
        background: '#0a0a0a',
        borderTop: focused ? '1px solid #444' : '1px solid #222',
        fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
        fontSize: '13px',
        zIndex: 50,
        display: 'flex',
        flexDirection: 'column',
      }}
      className="terminal-container"
    >
      {/* Output area */}
      <div
        ref={outputRef}
        style={{
          flex: 1,
          overflow: 'auto',
          padding: '8px 12px',
        }}
      >
        {output.map((line, i) => (
          <div
            key={i}
            style={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              color: line.type === 'error'
                ? '#f87171'
                : line.type === 'prompt'
                  ? '#e0e0e0'
                  : '#e0e0e0',
            }}
          >
            {line.type === 'output' ? formatOutput(line.content) : line.content}
          </div>
        ))}
      </div>

      {/* Input area */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '8px 12px',
          borderTop: '1px solid #1a1a1a',
        }}
      >
        <span style={{ color: '#666' }}>{prompt.slice(0, -3)}</span>
        <span style={{ color: '#e0e0e0' }}> $ </span>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          style={{
            flex: 1,
            background: 'transparent',
            border: 'none',
            outline: 'none',
            color: '#e0e0e0',
            fontFamily: 'inherit',
            fontSize: 'inherit',
            caretColor: '#e0e0e0',
          }}
          spellCheck={false}
          autoComplete="off"
          autoCorrect="off"
          autoCapitalize="off"
          aria-label="Terminal input"
        />
      </div>

      <style jsx global>{`
        .terminal-container::-webkit-scrollbar {
          width: 8px;
        }
        .terminal-container::-webkit-scrollbar-track {
          background: #0a0a0a;
        }
        .terminal-container::-webkit-scrollbar-thumb {
          background: #333;
          border-radius: 4px;
        }
        .terminal-container::-webkit-scrollbar-thumb:hover {
          background: #444;
        }
        @media (max-width: 767px) {
          .terminal-container {
            display: none !important;
          }
        }
      `}</style>
    </div>
  )
}

// Format output with colors for directories
function formatOutput(content: string): React.ReactNode {
  // Check if this looks like ls output (space-separated items)
  if (content.includes('/') && !content.includes('\n')) {
    const parts = content.split(/\s+/)
    return (
      <>
        {parts.map((part, i) => (
          <span key={i}>
            {i > 0 && '  '}
            <span style={{ color: part.endsWith('/') ? '#67e8f9' : '#e0e0e0' }}>
              {part}
            </span>
          </span>
        ))}
      </>
    )
  }
  return <>{content}</>
}

