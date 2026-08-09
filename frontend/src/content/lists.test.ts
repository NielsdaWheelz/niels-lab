/// <reference types="bun" />
import { expect, test } from 'bun:test'
import { lastWritten, type PillowList } from './lists'

test('lastWritten is the latest added date, not the last entry', () => {
  const list: PillowList = {
    slug: 'fixture',
    title: 'Fixture',
    entries: [
      { text: 'middle', added: '2026-03-02' },
      { text: 'latest', added: '2026-11-30' },
      { text: 'earliest', added: '2025-12-31' },
    ],
  }
  expect(lastWritten(list)).toBe('2026-11-30')
})

test('lastWritten of a single-entry list is that date', () => {
  const list: PillowList = {
    slug: 'fixture',
    title: 'Fixture',
    entries: [{ text: 'only', added: '2026-08-09' }],
  }
  expect(lastWritten(list)).toBe('2026-08-09')
})
