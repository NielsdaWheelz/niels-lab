import type { LedgerEvent, LedgerSource } from './types'

// 100 is the API maximum: branch churn eats a smaller window whole.
const EVENTS_URL =
  'https://api.github.com/users/NielsdaWheelz/events/public?per_page=100'
const DATE = /^\d{4}-\d{2}-\d{2}$/
const TRUNK = 'refs/heads/main'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function str(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

// The public events API carries no commit messages: a push is a ref and a
// head sha. A day's work on a repository is one row at the sha its trunk
// ended on — every merge lands there, and the branches, pull requests and
// deletions on the way are process, not shipping. Events arrive newest
// first, so the first trunk push seen for a day is where the day ended.
export function mapEvents(payload: unknown): LedgerEvent[] {
  if (!Array.isArray(payload)) {
    return []
  }

  const events: LedgerEvent[] = []
  const pushed = new Set<string>()

  for (const event of payload) {
    if (!isRecord(event)) {
      continue
    }

    const date = str(event.created_at).slice(0, 10)
    const repo = isRecord(event.repo) ? str(event.repo.name) : ''
    const body = isRecord(event.payload) ? event.payload : {}
    const head = str(body.head)
    const day = `${date} ${repo}`

    if (
      event.type !== 'PushEvent' ||
      str(body.ref) !== TRUNK ||
      !DATE.test(date) ||
      !repo.includes('/') ||
      !head ||
      pushed.has(day)
    ) {
      continue
    }

    pushed.add(day)
    events.push({
      date,
      kind: 'ship',
      text: `${repo.slice(repo.indexOf('/') + 1)} · main ${head.slice(0, 7)}`,
      href: `https://github.com/${repo}/commit/${head}`,
    })
  }

  return events
}

export const github: LedgerSource = {
  async get() {
    try {
      const response = await fetch(EVENTS_URL, {
        // The GitHub API rejects requests without a User-Agent.
        headers: {
          accept: 'application/vnd.github+json',
          'user-agent': 'nielseriknandal.com',
        },
        next: { revalidate: 86400 },
      })

      // The ledger must build without the network: an unreachable or
      // rate-limited API contributes no rows rather than failing the page.
      return response.ok ? mapEvents(await response.json()) : []
    } catch {
      return []
    }
  },
}
