/// <reference types="bun" />
import { expect, test } from 'bun:test'
import { AGE_STEPS, FAILURE_FACTOR, ageStep } from './inkAge'

// All dates are fixed YYYY-MM-DD literals measured against a single
// today ('2026-06-01') — no clock, per docs/testing-standards.md. Offsets
// below were computed once with `date -u -d '2026-06-01 -N days'`.

test('the published constants are the spec formula', () => {
  expect(AGE_STEPS).toEqual([0, 90, 180, 365, 730])
  expect(FAILURE_FACTOR).toBe(2)
})

const boundaries: [string, string, 0 | 1 | 2 | 3 | 4][] = [
  ['2026-06-01', '2026-06-01', 0], // same day
  ['2026-03-04', '2026-06-01', 0], // 89 days
  ['2026-03-03', '2026-06-01', 1], // 90 days
  ['2026-03-02', '2026-06-01', 1], // 91 days
  ['2025-12-04', '2026-06-01', 1], // 179 days
  ['2025-12-03', '2026-06-01', 2], // 180 days
  ['2025-06-02', '2026-06-01', 2], // 364 days
  ['2025-06-01', '2026-06-01', 3], // 365 days
  ['2024-06-02', '2026-06-01', 3], // 729 days
  ['2024-06-01', '2026-06-01', 4], // 730 days
  ['2023-09-05', '2026-06-01', 4], // 1000 days, far past
  ['2026-06-02', '2026-06-01', 0], // future date clamps to step 0
]

test.each(boundaries)(
  'ageStep(%s, %s) is step %i at the AGE_STEPS boundary',
  (rowDate, today, expected) => {
    expect(ageStep(rowDate, today)).toBe(expected)
  },
)

test('factor 2 doubles elapsed days before crossing a threshold', () => {
  // 44 days * 2 = 88, still under the 90 threshold.
  expect(ageStep('2026-04-18', '2026-06-01', FAILURE_FACTOR)).toBe(0)
})

test('factor 2 crosses a threshold at half the plain-factor distance', () => {
  // 45 days * 2 = 90, exactly the threshold.
  expect(ageStep('2026-04-17', '2026-06-01', FAILURE_FACTOR)).toBe(1)
})
