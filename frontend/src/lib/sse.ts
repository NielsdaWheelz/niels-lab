/**
 * Streaming client for NDJSON (newline-delimited JSON) responses
 * Handles concatenated JSON objects like: {...}{...}{...}
 */

export type StreamToken = {
  step: number
  token_id: number
  token_text: string
  token_display: string
}

export async function* fetchStream(
  url: string,
  init: RequestInit,
): AsyncGenerator<StreamToken> {
  const res = await fetch(url, init)

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`HTTP ${res.status}: ${text}`)
  }

  if (!res.body) {
    throw new Error('No response body')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()

  let buf = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buf += decoder.decode(value, { stream: true })

    // Extract complete JSON objects from buffer
    // Pattern: find each {...} object
    while (true) {
      // Find the start of a JSON object
      const startIdx = buf.indexOf('{')
      if (startIdx === -1) {
        buf = '' // No objects, clear buffer
        break
      }

      // Find matching closing brace
      let depth = 0
      let inString = false
      let escapeNext = false
      let endIdx = -1

      for (let i = startIdx; i < buf.length; i++) {
        const char = buf[i]

        if (escapeNext) {
          escapeNext = false
          continue
        }

        if (char === '\\' && inString) {
          escapeNext = true
          continue
        }

        if (char === '"') {
          inString = !inString
          continue
        }

        if (inString) continue

        if (char === '{') {
          depth++
        } else if (char === '}') {
          depth--
          if (depth === 0) {
            endIdx = i
            break
          }
        }
      }

      if (endIdx === -1) {
        // Incomplete object, wait for more data
        // Keep from startIdx onwards
        buf = buf.slice(startIdx)
        break
      }

      // Extract and parse the complete object
      const jsonStr = buf.slice(startIdx, endIdx + 1)
      buf = buf.slice(endIdx + 1)

      try {
        const parsed = JSON.parse(jsonStr)
        // Only yield if it has the expected fields
        if (typeof parsed.token_text === 'string') {
          yield parsed as StreamToken
        }
      } catch (e) {
        console.warn('Failed to parse JSON:', jsonStr, e)
      }
    }
  }

  // Process any remaining complete objects in buffer
  while (buf.includes('{') && buf.includes('}')) {
    const startIdx = buf.indexOf('{')
    let depth = 0
    let inString = false
    let escapeNext = false
    let endIdx = -1

    for (let i = startIdx; i < buf.length; i++) {
      const char = buf[i]
      if (escapeNext) {
        escapeNext = false
        continue
      }
      if (char === '\\' && inString) {
        escapeNext = true
        continue
      }
      if (char === '"') {
        inString = !inString
        continue
      }
      if (inString) continue
      if (char === '{') depth++
      else if (char === '}') {
        depth--
        if (depth === 0) {
          endIdx = i
          break
        }
      }
    }

    if (endIdx === -1) break

    const jsonStr = buf.slice(startIdx, endIdx + 1)
    buf = buf.slice(endIdx + 1)

    try {
      const parsed = JSON.parse(jsonStr)
      if (typeof parsed.token_text === 'string') {
        yield parsed as StreamToken
      }
    } catch {
      // Skip malformed
    }
  }
}
