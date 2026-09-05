const DAY_MS = 86_400_000

// Days -> ledger ink-age thresholds; each maps to a --age-N token in
// globals.css (spec §5.4/§6). Step 0 covers same-day through 89 days.
export const AGE_STEPS = [0, 90, 180, 365, 730]

// A struck failure dries twice as fast as its cohort. Its lesson is exempt
// from this scale entirely (rendered in full celadon at any age) — that
// exemption is a rendering-layer decision, not modeled here.
export const FAILURE_FACTOR = 2

// rowDate/today are YYYY-MM-DD. Effective age scales elapsed days by
// factor; the step is the largest AGE_STEPS threshold at or under it.
export function ageStep(
  rowDate: string,
  today: string,
  factor = 1,
): 0 | 1 | 2 | 3 | 4 {
  const days =
    (Date.parse(`${today}T00:00:00Z`) - Date.parse(`${rowDate}T00:00:00Z`)) /
    DAY_MS
  const effectiveAge = Math.max(0, days * factor)
  return (AGE_STEPS.filter((threshold) => threshold <= effectiveAge).length -
    1) as 0 | 1 | 2 | 3 | 4
}
