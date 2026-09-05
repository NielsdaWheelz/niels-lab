export type HourBand =
  | 'small-hours'
  | 'dawn'
  | 'morning'
  | 'afternoon'
  | 'dusk'
  | 'evening'

const laHour = new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/Los_Angeles',
  hourCycle: 'h23',
  hour: 'numeric',
})

// Bands are the half-open America/Los_Angeles wall-clock ranges from spec
// §6. Intl resolves the LA offset for the given instant, so this is
// DST-correct by construction — no fixed offset, no arithmetic to get wrong
// across the spring-forward/fall-back transitions.
export function band(date: Date): HourBand {
  const hour = Number(laHour.format(date))
  if (hour < 5) return 'small-hours'
  if (hour < 8) return 'dawn'
  if (hour < 12) return 'morning'
  if (hour < 17) return 'afternoon'
  if (hour < 20) return 'dusk'
  return 'evening'
}
