'use client'

import React, { useRef, useEffect, useState, KeyboardEvent } from 'react'
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
  const [expanded, setExpanded] = useState(false) // for mobile
  const [isMobile, setIsMobile] = useState(false)

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

  // Detect mobile and handle resize
  useEffect(() => {
    setMounted(true)
    
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 640)
    }
    
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
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
        if (isMobile) {
          setExpanded(true)
        }
        setTimeout(() => inputRef.current?.focus(), 50)
      }
    }

    window.addEventListener('keydown', handleGlobalKeyDown)
    return () => window.removeEventListener('keydown', handleGlobalKeyDown)
  }, [isMobile])

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
        if (isMobile) {
          setExpanded(false)
        }
        break
    }
  }

  const handleMobileToggle = () => {
    if (isMobile) {
      setExpanded(!expanded)
      if (!expanded) {
        setTimeout(() => inputRef.current?.focus(), 50)
      }
    }
  }

  if (!mounted) {
    return (
      <div
        style={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          height: isMobile ? '48px' : '140px',
          background: '#0a0a0a',
          borderTop: '1px solid #222',
          zIndex: 50,
        }}
        className="terminal-container"
      />
    )
  }

  // Mobile collapsed state
  const mobileCollapsedHeight = 48
  const mobileExpandedHeight = 200
  const desktopHeight = 140

  const terminalHeight = isMobile 
    ? (expanded ? mobileExpandedHeight : mobileCollapsedHeight)
    : desktopHeight

  return (
    <div
      style={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        height: `${terminalHeight}px`,
        background: '#0a0a0a',
        borderTop: focused ? '1px solid #444' : '1px solid #222',
        fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
        fontSize: isMobile ? '12px' : '13px',
        zIndex: 50,
        display: 'flex',
        flexDirection: 'column',
        transition: 'height 0.2s ease',
      }}
      className="terminal-container"
    >
      {/* Mobile header bar */}
      {isMobile && (
        <div
          onClick={handleMobileToggle}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '8px 12px',
            cursor: 'pointer',
            borderBottom: expanded ? '1px solid #1a1a1a' : 'none',
            userSelect: 'none',
          }}
        >
          <span style={{ color: '#666' }}>
            {expanded ? '▼' : '▲'} terminal
          </span>
          <span style={{ color: '#444', fontSize: '10px' }}>
            {expanded ? 'tap to collapse' : 'tap to expand'}
          </span>
        </div>
      )}

      {/* Output area - hidden on mobile when collapsed */}
      {(!isMobile || expanded) && (
        <div
          ref={outputRef}
          style={{
            flex: 1,
            overflow: 'auto',
            padding: isMobile ? '6px 10px' : '8px 12px',
            display: isMobile && !expanded ? 'none' : 'block',
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
                  : '#e0e0e0',
                fontSize: isMobile ? '11px' : '13px',
                lineHeight: 1.4,
              }}
            >
              {line.type === 'output' ? formatOutput(line.content) : line.content}
            </div>
          ))}
        </div>
      )}

      {/* Input area */}
      {(!isMobile || expanded) && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            padding: isMobile ? '6px 10px' : '8px 12px',
            borderTop: '1px solid #1a1a1a',
          }}
        >
          <span style={{ color: '#666', fontSize: isMobile ? '11px' : '13px' }}>
            {prompt.slice(0, -3)}
          </span>
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
              fontSize: isMobile ? '12px' : '13px',
              caretColor: '#e0e0e0',
              minWidth: 0, // allow shrinking
            }}
            spellCheck={false}
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            aria-label="Terminal input"
          />
        </div>
      )}

      <style jsx global>{`
        .terminal-container::-webkit-scrollbar {
          width: 6px;
        }
        .terminal-container::-webkit-scrollbar-track {
          background: #0a0a0a;
        }
        .terminal-container::-webkit-scrollbar-thumb {
          background: #333;
          border-radius: 3px;
        }
        .terminal-container::-webkit-scrollbar-thumb:hover {
          background: #444;
        }
      `}</style>
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
