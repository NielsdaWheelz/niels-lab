import { github } from './github'
import { manual } from './manual'
import type { LedgerEvent } from './types'

const QUIET_DAYS = 14
const DAY_MS = 86_400_000

// Batches arrive in source priority order: the first source to claim a
// (date, text) keeps the row, so a hand-written row with its lesson and
// href survives the same day's automatic one.
export function mergeEvents(batches: LedgerEvent[][]): LedgerEvent[] {
  const seen = new Set<string>()
  const merged: LedgerEvent[] = []

  for (const event of batches.flat()) {
    const key = `${event.date} ${event.text}`
    if (seen.has(key)) {
      continue
    }
    seen.add(key)
    merged.push(event)
  }

  // sort is stable: rows sharing a date keep source order.
  return merged.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0))
}

// Quiet for a fortnight means the page reports the absence instead of
// pretending the ledger is current. An empty ledger is quiet by definition.
export function isStale(events: LedgerEvent[], today: string): boolean {
  const latest = events.reduce(
    (newest, event) => (event.date > newest ? event.date : newest),
    '',
  )

  if (!latest) {
    return true
  }

  const quiet =
    Date.parse(`${today}T00:00:00Z`) - Date.parse(`${latest}T00:00:00Z`)
  return quiet >= QUIET_DAYS * DAY_MS
}

export async function getLedger(): Promise<LedgerEvent[]> {
  return mergeEvents(await Promise.all([manual.get(), github.get()]))
}
