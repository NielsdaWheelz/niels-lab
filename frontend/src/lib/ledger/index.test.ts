/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { isStale, mergeEvents } from './index'
import type { LedgerEvent } from './types'

const manual: LedgerEvent[] = [
  { date: '2026-07-20', kind: 'ship', text: 'niels-lab · main 03d5b44' },
  {
    date: '2026-08-02',
    kind: 'train',
    text: 'SQ 3×3 @385 · third set moved like the first',
  },
]

const github: LedgerEvent[] = [
  { date: '2026-08-09', kind: 'ship', text: 'niels-lab · main 4fc8401' },
  {
    date: '2026-07-20',
    kind: 'ship',
    text: 'niels-lab · main 03d5b44',
    href: 'https://github.com/NielsdaWheelz/niels-lab/commit/03d5b44',
  },
]

describe('mergeEvents', () => {
  test('reads newest first across sources', () => {
    expect(mergeEvents([manual, github]).map((event) => event.date)).toEqual([
      '2026-08-09',
      '2026-08-02',
      '2026-07-20',
    ])
  })

  test('keeps the hand-written row when two sources report the same day and text', () => {
    const merged = mergeEvents([manual, github])
    const shared = merged.filter((event) => event.date === '2026-07-20')

    expect(shared).toEqual([
      { date: '2026-07-20', kind: 'ship', text: 'niels-lab · main 03d5b44' },
    ])
  })

  test('keeps two rows written on one day', () => {
    const twice: LedgerEvent[] = [
      { date: '2026-07-20', kind: 'ship', text: 'first' },
      { date: '2026-07-20', kind: 'read', text: 'second' },
    ]

    expect(mergeEvents([twice])).toEqual(twice)
  })
})

describe('isStale', () => {
  const on = (date: string): LedgerEvent[] => [
    { date, kind: 'train', text: 'SQ 5×5 @335 · pickup 90 min' },
  ]

  test('is quiet at thirteen days', () => {
    expect(isStale(on('2026-07-27'), '2026-08-09')).toBe(false)
  })

  test('speaks at fourteen days', () => {
    expect(isStale(on('2026-07-26'), '2026-08-09')).toBe(true)
  })

  test('stays spoken at fifteen days', () => {
    expect(isStale(on('2026-07-25'), '2026-08-09')).toBe(true)
  })

  test('measures from the newest row, not the last one given', () => {
    expect(
      isStale([...on('2026-07-01'), ...on('2026-08-08')], '2026-08-09'),
    ).toBe(false)
  })

  test('an empty ledger is a quiet one', () => {
    expect(isStale([], '2026-08-09')).toBe(true)
  })
})
