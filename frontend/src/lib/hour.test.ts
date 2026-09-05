/// <reference types="bun" />
import { expect, test } from 'bun:test'
import { band, type HourBand } from './hour'

// All fixtures are fixed UTC instants — no clock, no local timezone
// dependency (per docs/testing-standards.md). Boundary instants are drawn
// from 2026-01-15/16, a winter (PST, UTC-8) date well clear of any DST
// transition, so each expected LA time is plain UTC-8 arithmetic.

const boundaries: [string, HourBand][] = [
  ['2026-01-15T12:59:00Z', 'small-hours'], // 04:59 LA
  ['2026-01-15T13:00:00Z', 'dawn'], // 05:00 LA
  ['2026-01-15T15:59:00Z', 'dawn'], // 07:59 LA
  ['2026-01-15T16:00:00Z', 'morning'], // 08:00 LA
  ['2026-01-15T19:59:00Z', 'morning'], // 11:59 LA
  ['2026-01-15T20:00:00Z', 'afternoon'], // 12:00 LA
  ['2026-01-16T00:59:00Z', 'afternoon'], // 16:59 LA
  ['2026-01-16T01:00:00Z', 'dusk'], // 17:00 LA
  ['2026-01-16T03:59:00Z', 'dusk'], // 19:59 LA
  ['2026-01-16T04:00:00Z', 'evening'], // 20:00 LA
  ['2026-01-16T07:59:00Z', 'evening'], // 23:59 LA
  ['2026-01-16T08:00:00Z', 'small-hours'], // 00:00 LA (next day)
]

test.each(boundaries)(
  'band(%s) is %s at every LA hour boundary',
  (iso, expected) => {
    expect(band(new Date(iso))).toBe(expected)
  },
)

test('band is correct just after spring-forward, when LA is UTC-7 not UTC-8', () => {
  // 2026-03-08: LA clocks skip 02:00-03:00 PST -> PDT at 10:00Z. At 15:00Z
  // the correct (UTC-7) LA time is 08:00 (morning); a fixed UTC-8 offset
  // would read 07:00 and misreport dawn.
  expect(band(new Date('2026-03-08T15:00:00Z'))).toBe('morning')
})

test('band is correct just after fall-back, when LA is UTC-8 not UTC-7', () => {
  // 2026-11-01: LA clocks repeat 01:00-02:00 as PDT -> PST falls back at
  // 09:00Z. At 2026-11-02T00:00Z the correct (UTC-8) LA time is 16:00
  // (afternoon); a fixed UTC-7 offset would read 17:00 and misreport dusk.
  expect(band(new Date('2026-11-02T00:00:00Z'))).toBe('afternoon')
})
