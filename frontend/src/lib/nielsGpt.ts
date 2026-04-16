/**
 * Client for niels-gpt-api streaming chat endpoint
 */

import { fetchStream } from './sse'
import { FileSystem } from './filesystem'

export type Role = 'system' | 'user' | 'assistant'

export type Message = {
  role: Role
  content: string
}

const API_BASE = process.env.NEXT_PUBLIC_NIELS_GPT_API_BASE || 'https://niels-gpt-api.onrender.com'

// System prompt for the assistant
const SYSTEM_PROMPT = `you are the terminal assistant for niels erik nandal's portfolio website.
the terminal's canonical top-level paths are /projects, /writing, and /cv.
refer to him as niels unless the user asks for the full name.
speak casually and concisely. keep responses brief unless asked for detail.
you can suggest using slash commands like /search, /whois, /ls, /cd for navigation.
don't invent facts about niels - stick to what's in the terminal filesystem or admit you don't know.`

// Conversation history - maintained across chat sessions
const conversationHistory: Message[] = [
  { role: 'system', content: SYSTEM_PROMPT }
]

// Stop sequences - patterns that indicate the model is generating dialogue markers
const STOP_PATTERNS = [
  /\n*<dialogue>/i,
  /\n*<\/dialogue>/i,
  /\n*system:/i,
  /\n*user:/i,
  /\n*assistant:/i,
  /\n*<\|/,  // Common special tokens like <|end|>, <|user|>, etc.
]

/**
 * Clean up response by removing trailing dialogue markers and artifacts
 */
function cleanResponse(text: string): string {
  let cleaned = text

  // Find the earliest stop pattern and truncate there
  let earliestIdx = cleaned.length
  for (const pattern of STOP_PATTERNS) {
    const match = cleaned.match(pattern)
    if (match && match.index !== undefined && match.index < earliestIdx) {
      earliestIdx = match.index
    }
  }

  cleaned = cleaned.slice(0, earliestIdx)

  // Trim trailing whitespace and newlines
  cleaned = cleaned.trimEnd()

  return cleaned
}

/**
 * Streaming chat handler compatible with useTerminal's ChatHandler type
 */
export async function handleChat(
  userMessage: string,
  _context: { cwd: string; fs: FileSystem },
  callbacks: {
    onToken: (token: string) => void
    onDone: () => void
    onError: (error: Error) => void
    signal: AbortSignal
  }
): Promise<void> {
  const { onToken, onDone, onError, signal } = callbacks

  // Add user message to history
  conversationHistory.push({ role: 'user', content: userMessage })

  const url = `${API_BASE}/chat/stream`

  const body = JSON.stringify({
    messages: conversationHistory,
    max_new_tokens: 256,
    temperature: 0.7,
    trace_layer: 0,  // Required by backend
  })

  let assistantResponse = ''

  try {
    let shouldStop = false

    for await (const token of fetchStream(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      signal,
    })) {
      if (shouldStop) break

      // Each token has { step, token_id, token_text, token_display }
      // Skip if token_text is missing or undefined
      if (token.token_text === undefined || token.token_text === null) {
        console.warn('Received token without token_text:', token)
        continue
      }

      const text = token.token_text
      assistantResponse += text

      // Check if we've hit a stop pattern - don't emit tokens after stop
      for (const pattern of STOP_PATTERNS) {
        if (pattern.test(assistantResponse)) {
          shouldStop = true
          // Clean response before final emit
          assistantResponse = cleanResponse(assistantResponse)
          break
        }
      }

      if (!shouldStop) {
        onToken(text)
      }
    }

    // Stream finished - clean up and add response to history
    // Remove any trailing dialogue/system markers the model might generate
    assistantResponse = cleanResponse(assistantResponse)

    if (assistantResponse) {
      conversationHistory.push({ role: 'assistant', content: assistantResponse })
      // Trim history if too long (keep system + last 20 messages)
      if (conversationHistory.length > 21) {
        conversationHistory.splice(1, conversationHistory.length - 21)
      }
    }
    onDone()
  } catch (err) {
    if ((err as Error).name === 'AbortError') {
      // Request cancelled - still save partial response if any
      if (assistantResponse) {
        conversationHistory.push({ role: 'assistant', content: assistantResponse + ' [cancelled]' })
      }
      onDone()
      return
    }
    // Remove the user message from history on error
    conversationHistory.pop()
    onError(err as Error)
  }
}

/**
 * Clear conversation history (useful for /clear command integration)
 */
export function clearConversationHistory(): void {
  conversationHistory.length = 1 // Keep only system prompt
}
