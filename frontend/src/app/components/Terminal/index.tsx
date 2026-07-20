'use client'

import React, {
  useRef,
  useEffect,
  useState,
  useSyncExternalStore,
  KeyboardEvent,
  useMemo,
} from 'react'
import Link from 'next/link'
import { FileSystem } from '@/lib/filesystem'
import { useTerminal, OutputLine, ChatHandler } from './useTerminal'
import { TextReveal } from '../TextReveal'
import { githubUrl, linkedinUrl } from '@/app/site'
import { getTheme, Theme, THEME_CHANGE_EVENT } from '@/lib/theme'
import { TERMINAL_OPEN_EVENT } from '@/lib/terminal'

interface TerminalProps {
  filesystem: FileSystem
  onChat?: ChatHandler
  initiallyExpanded?: boolean
}

// Seeded PRNG for deterministic randomness (avoid hydration mismatch)
function seededRandom(seed: number) {
  return () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return seed / 0x7fffffff
  }
}

// Types for special command outputs
type SearchResult = {
  __type: 'search'
  query: string
  results: Array<{
    title: string
    path: string
    snippet: string
    lineNum?: number
  }>
}

type WhoisOutput = {
  __type: 'whois'
}

type VisualizeOutput = {
  __type: 'visualize'
  seed: number
}

type NeofetchOutput = {
  __type: 'neofetch'
}

type SpecialOutput =
  | SearchResult
  | WhoisOutput
  | VisualizeOutput
  | NeofetchOutput

function isSpecialOutput(content: unknown): content is SpecialOutput {
  return typeof content === 'object' && content !== null && '__type' in content
}

// Highlight search matches in text
function highlightMatches(text: string, query: string): React.ReactNode {
  if (!query) return text

  const lowerText = text.toLowerCase()
  const lowerQuery = query.toLowerCase()
  const parts: React.ReactNode[] = []
  let lastIndex = 0
  let index = lowerText.indexOf(lowerQuery)

  while (index !== -1) {
    if (index > lastIndex) {
      parts.push(text.slice(lastIndex, index))
    }
    parts.push(
      <mark key={index} className="terminal-highlight">
        {text.slice(index, index + query.length)}
      </mark>,
    )
    lastIndex = index + query.length
    index = lowerText.indexOf(lowerQuery, lastIndex)
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }

  return parts.length > 0 ? <>{parts}</> : text
}

// Linkify paths and highlight commands in string output
function formatStringOutput(content: string): React.ReactNode {
  // Paths to linkify
  const pathPattern = /(\/(?:projects|writing|cv)(?:\/[^\s,]+)*)/g
  // Commands to highlight
  const commandNames = [
    'help',
    'search',
    'grep',
    'whois',
    'visualize',
    'theme',
    'fortune',
    'neofetch',
    'sudo',
    'pwd',
    'ls',
    'cd',
    'cat',
    'clear',
  ]
  // Backticked words
  const backtickPattern = /`([^`]+)`/g

  const lines = content.split('\n')

  return (
    <>
      {lines.map((line, lineIdx) => {
        const parts: React.ReactNode[] = []
        let partKey = 0

        // Process paths first
        const pathMatches = [...line.matchAll(pathPattern)]
        if (pathMatches.length > 0) {
          let lastIdx = 0
          for (const match of pathMatches) {
            const before = line.slice(lastIdx, match.index)
            if (before)
              parts.push(
                <span key={partKey++}>
                  {processCommandsAndBackticks(
                    before,
                    commandNames,
                    backtickPattern,
                  )}
                </span>,
              )

            const path = match[0]
            parts.push(
              <Link key={partKey++} href={path} className="terminal-link">
                {path}
              </Link>,
            )
            lastIdx = (match.index || 0) + path.length
          }
          const after = line.slice(lastIdx)
          if (after)
            parts.push(
              <span key={partKey++}>
                {processCommandsAndBackticks(
                  after,
                  commandNames,
                  backtickPattern,
                )}
              </span>,
            )
        } else {
          parts.push(
            <span key={partKey++}>
              {processCommandsAndBackticks(line, commandNames, backtickPattern)}
            </span>,
          )
        }

        return (
          <React.Fragment key={lineIdx}>
            {lineIdx > 0 && '\n'}
            {parts}
          </React.Fragment>
        )
      })}
    </>
  )
}

function processCommandsAndBackticks(
  text: string,
  commandNames: string[],
  backtickPattern: RegExp,
): React.ReactNode {
  // First handle backticks
  const backtickMatches = [...text.matchAll(backtickPattern)]
  if (backtickMatches.length > 0) {
    const parts: React.ReactNode[] = []
    let lastIdx = 0
    let partKey = 0

    for (const match of backtickMatches) {
      const before = text.slice(lastIdx, match.index)
      if (before)
        parts.push(
          <span key={partKey++}>
            {highlightCommands(before, commandNames)}
          </span>,
        )

      parts.push(
        <span key={partKey++} className="terminal-command">
          {match[1]}
        </span>,
      )
      lastIdx = (match.index || 0) + match[0].length
    }
    const after = text.slice(lastIdx)
    if (after)
      parts.push(
        <span key={partKey++}>{highlightCommands(after, commandNames)}</span>,
      )
    return <>{parts}</>
  }

  return highlightCommands(text, commandNames)
}

function highlightCommands(
  text: string,
  commandNames: string[],
): React.ReactNode {
  // Create pattern for standalone command words
  const pattern = new RegExp(`\\b(${commandNames.join('|')})\\b`, 'g')
  const matches = [...text.matchAll(pattern)]

  if (matches.length === 0) return text

  const parts: React.ReactNode[] = []
  let lastIdx = 0
  let partKey = 0

  for (const match of matches) {
    const before = text.slice(lastIdx, match.index)
    if (before) parts.push(before)

    parts.push(
      <span key={partKey++} className="terminal-command">
        {match[0]}
      </span>,
    )
    lastIdx = (match.index || 0) + match[0].length
  }
  const after = text.slice(lastIdx)
  if (after) parts.push(after)

  return <>{parts}</>
}

// Search results component
function SearchResults({ data }: { data: SearchResult }) {
  return (
    <div className="terminal-results">
      {data.results.map((result, idx) => (
        <div key={idx} className="terminal-result-card">
          <div className="terminal-result-title">
            <Link href={result.path} className="terminal-link">
              {result.title}
            </Link>
          </div>
          <div className="terminal-result-path">{result.path}</div>
          <div className="terminal-result-snippet">
            {result.lineNum && (
              <span className="terminal-line-num">{result.lineNum}: </span>
            )}
            {highlightMatches(result.snippet, data.query)}
          </div>
        </div>
      ))}
    </div>
  )
}

// Whois persona card
function WhoisCard() {
  return (
    <div className="terminal-card">
      <div className="terminal-card-header">
        <span className="terminal-card-name">niels</span>
        <span className="terminal-card-role">software engineer</span>
      </div>
      <p className="terminal-card-bio">
        engineer building ai products, systems, and readable software.
      </p>
      <div className="terminal-card-facts">
        <div className="terminal-card-fact">
          <span className="terminal-card-label">focus</span>
          <span>ai products & backend systems</span>
        </div>
        <div className="terminal-card-fact">
          <span className="terminal-card-label">stack</span>
          <span>python, typescript, sql</span>
        </div>
        <div className="terminal-card-fact">
          <span className="terminal-card-label">interests</span>
          <span>product loops, ml systems, maintainable code</span>
        </div>
      </div>
      <div className="terminal-card-links">
        <a
          href={githubUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="terminal-link"
        >
          github
        </a>
        <span className="terminal-card-sep">|</span>
        <a
          href={linkedinUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="terminal-link"
        >
          linkedin
        </a>
        <span className="terminal-card-sep">|</span>
        <Link href="/cv" className="terminal-link">
          cv
        </Link>
      </div>
      <div className="terminal-card-stats">
        <span>uptime: </span>
        <span className="terminal-stat-bar">[##########]</span>
        <span> 99.9%</span>
      </div>
    </div>
  )
}

function subscribeToTheme(onStoreChange: () => void) {
  window.addEventListener(THEME_CHANGE_EVENT, onStoreChange)
  return () => window.removeEventListener(THEME_CHANGE_EVENT, onStoreChange)
}

function getServerTheme(): Theme {
  return 'light'
}

// computed at module load; the card only renders client-side on demand
const UPTIME_YEARS = (
  (Date.now() - new Date('2024-01-01T00:00:00Z').getTime()) /
  (365.25 * 24 * 60 * 60 * 1000)
).toFixed(1)

// ASCII "N" logo for neofetch
const NEOFETCH_ASCII = [
  ' _   _ ',
  '| \\ | |',
  '|  \\| |',
  '| |\\  |',
  '|_| \\_|',
]

// Neofetch-style system info card
function NeofetchCard() {
  const theme = useSyncExternalStore(subscribeToTheme, getTheme, getServerTheme)

  const uptime = UPTIME_YEARS

  const palette: Array<[string, string]> = [
    ['terracotta', 'var(--color-terracotta)'],
    ['sage', 'var(--color-sage)'],
    ['gold', 'var(--color-gold)'],
    ['text', 'var(--color-text)'],
    ['muted', 'var(--color-text-muted)'],
    ['border', 'var(--color-border)'],
  ]

  return (
    <div className="terminal-card">
      <div className="terminal-card-header">
        <span className="terminal-card-name">niels@lab</span>
        <span className="terminal-card-role">neofetch</span>
      </div>
      <div style={{ display: 'flex', gap: '1.25rem', marginBottom: '0.75rem' }}>
        <pre
          style={{
            margin: 0,
            lineHeight: 1.25,
            fontSize: '0.85rem',
            color: 'var(--color-sage)',
          }}
          aria-hidden="true"
        >
          {NEOFETCH_ASCII.join('\n')}
        </pre>
        <div className="terminal-card-facts" style={{ marginBottom: 0 }}>
          <div className="terminal-card-fact">
            <span className="terminal-card-label">host</span>
            <span>niels-lab (sketchbook edition)</span>
          </div>
          <div className="terminal-card-fact">
            <span className="terminal-card-label">stack</span>
            <span>next 16 / react 19 / bun</span>
          </div>
          <div className="terminal-card-fact">
            <span className="terminal-card-label">theme</span>
            <span>{theme}</span>
          </div>
          <div className="terminal-card-fact">
            <span className="terminal-card-label">uptime</span>
            <span>{uptime} years</span>
          </div>
          <div className="terminal-card-fact">
            <span className="terminal-card-label">shell</span>
            <span>niels-sh</span>
          </div>
        </div>
      </div>
      <div
        className="terminal-card-stats"
        style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}
      >
        {palette.map(([name, color]) => (
          <span
            key={name}
            title={name}
            aria-hidden="true"
            style={{
              display: 'inline-block',
              width: '13px',
              height: '13px',
              borderRadius: '3px',
              background: color,
              border: '1px solid var(--color-border)',
            }}
          />
        ))}
      </div>
    </div>
  )
}

// Visualize SVG scribble
function VisualizeScribble({ seed }: { seed: number }) {
  const paths = useMemo(() => {
    const rand = seededRandom(seed)
    const colors = [
      'var(--color-terracotta)',
      'var(--color-sage)',
      'var(--color-gold)',
      'var(--color-text)',
    ]
    const elements: React.ReactNode[] = []

    // Generate 3-5 wobbly lines
    const lineCount = Math.floor(rand() * 3) + 3
    for (let i = 0; i < lineCount; i++) {
      const startX = rand() * 40 + 10
      const startY = rand() * 60 + 20
      const points: string[] = [`M ${startX} ${startY}`]

      // 2-4 curve segments
      const segments = Math.floor(rand() * 3) + 2
      for (let j = 0; j < segments; j++) {
        const cp1x = startX + (j + 1) * 60 + (rand() - 0.5) * 30
        const cp1y = startY + (rand() - 0.5) * 40
        const endX = startX + (j + 1) * 80 + (rand() - 0.5) * 20
        const endY = startY + (rand() - 0.5) * 50
        points.push(`Q ${cp1x} ${cp1y} ${endX} ${endY}`)
      }

      elements.push(
        <path
          key={`line-${i}`}
          d={points.join(' ')}
          stroke={colors[i % colors.length]}
          strokeWidth={1.5 + rand()}
          strokeLinecap="round"
          fill="none"
          opacity={0.6 + rand() * 0.4}
        />,
      )
    }

    // Add some dots
    const dotCount = Math.floor(rand() * 6) + 4
    for (let i = 0; i < dotCount; i++) {
      elements.push(
        <circle
          key={`dot-${i}`}
          cx={rand() * 320 + 20}
          cy={rand() * 80 + 10}
          r={2 + rand() * 3}
          fill={colors[Math.floor(rand() * colors.length)]}
          opacity={0.4 + rand() * 0.5}
        />,
      )
    }

    return elements
  }, [seed])

  return (
    <svg
      className="terminal-scribble"
      viewBox="0 0 360 100"
      fill="none"
      aria-hidden="true"
    >
      {paths}
    </svg>
  )
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

function OutputLineContent({
  line,
  onTypewriterComplete,
}: {
  line: OutputLine
  onTypewriterComplete?: (id: string) => void
}) {
  if (line.type === 'thinking') {
    return <ThinkingIndicator />
  }

  const content = line.content

  // Check for special structured outputs
  if (isSpecialOutput(content)) {
    switch (content.__type) {
      case 'search':
        return <SearchResults data={content} />
      case 'whois':
        return <WhoisCard />
      case 'visualize':
        return <VisualizeScribble seed={content.seed} />
      case 'neofetch':
        return <NeofetchCard />
    }
  }

  // String output with potential typewriter effect
  if (typeof content === 'string') {
    if (line.type === 'output') {
      // Check if this looks like ls output (directories need special formatting)
      if (
        content.includes('/') &&
        !content.includes('\n') &&
        content.split(/\s+/).length <= 20
      ) {
        const parts = content.split(/\s+/).filter(Boolean)
        if (line.isTyping) {
          return (
            <TextReveal
              text={content}
              speed={30}
              showCursor={true}
              onComplete={() => onTypewriterComplete?.(line.id)}
            />
          )
        }
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

      // Regular string output with typewriter
      if (line.isTyping) {
        return (
          <TextReveal
            text={content}
            speed={30}
            showCursor={true}
            onComplete={() => onTypewriterComplete?.(line.id)}
          />
        )
      }

      // Completed typewriter or no typewriter - apply formatting
      return <>{formatStringOutput(content)}</>
    }

    // Error or prompt - no typewriter, just render
    return <>{content}</>
  }

  // React node output - render directly (no typewriter)
  return <>{content}</>
}

export function Terminal({
  filesystem,
  onChat,
  initiallyExpanded = false,
}: TerminalProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const outputRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const [focused, setFocused] = useState(false)
  const [mounted, setMounted] = useState(false)
  const [expanded, setExpanded] = useState(initiallyExpanded)

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
    status,
    onTypewriterComplete,
    cancelRequest,
  } = useTerminal(filesystem, onChat)

  useEffect(() => {
    const frameId = requestAnimationFrame(() => {
      setMounted(true)
      if (initiallyExpanded) {
        setTimeout(() => inputRef.current?.focus(), 50)
      }
    })

    return () => cancelAnimationFrame(frameId)
  }, [initiallyExpanded])

  // auto-scroll output
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [output])

  // global ctrl+k handler
  useEffect(() => {
    const openTerminal = () => {
      setExpanded(true)
      setTimeout(() => inputRef.current?.focus(), 50)
    }

    const handleGlobalKeyDown = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'k' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        openTerminal()
      }
    }

    window.addEventListener('keydown', handleGlobalKeyDown)
    window.addEventListener(TERMINAL_OPEN_EVENT, openTerminal)
    return () => {
      window.removeEventListener('keydown', handleGlobalKeyDown)
      window.removeEventListener(TERMINAL_OPEN_EVENT, openTerminal)
    }
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
        if (isThinking) {
          // Cancel ongoing LLM request
          cancelRequest()
        } else {
          inputRef.current?.blur()
          setExpanded(false)
        }
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
  // Map status values to display text
  const statusText =
    status === 'idle' ? 'ready.' : status === 'error' ? 'error.' : status

  return (
    <div
      ref={panelRef}
      className={`assistant-panel ${expanded ? 'expanded' : ''} ${focused ? 'focused' : ''}`}
    >
      {/* Collapsed state: just the input bar */}
      {!expanded && (
        <button
          type="button"
          className="assistant-collapsed"
          onClick={handleToggle}
          aria-expanded="false"
          aria-label="Open the interactive site terminal"
        >
          <span className="assistant-prompt-hint">~</span>
          <span className="assistant-placeholder">
            chat or /help for commands
          </span>
          <kbd className="assistant-shortcut">^K</kbd>
        </button>
      )}

      {/* Expanded state: output + input + status */}
      {expanded && (
        <>
          {/* Output area */}
          {hasOutput && (
            <div
              ref={outputRef}
              className="assistant-output"
              aria-live="polite"
              aria-label="Terminal output"
            >
              {output.map((line) => (
                <div
                  key={line.id}
                  className={`assistant-line assistant-line--${line.type}`}
                >
                  <OutputLineContent
                    line={line}
                    onTypewriterComplete={onTypewriterComplete}
                  />
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
              placeholder={
                hasOutput ? '' : 'chat or type /help for commands...'
              }
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

          {/* Status line */}
          <div className="assistant-status" role="status">
            {statusText}
          </div>
        </>
      )}
    </div>
  )
}
