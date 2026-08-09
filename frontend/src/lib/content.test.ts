/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { parseFrontmatter } from './content'

// Shape of a real post: a title carrying its own colon, quoted values,
// optional keys, and a body that must survive intact.
const post = `---
title: 'Suno Session Lab: turning clip generation into a navigable space'
publishedAt: '2025-12-11'
summary: 'Turns music generation into a navigable space'
repoUrl: 'https://github.com/NielsdaWheelz/suno-demo'
---

## why i built this

Most music AI feels like a slot machine. You pull the lever and you guess.
`

describe('parseFrontmatter', () => {
  test('reads one key per line, strips quotes, keeps colons in the value', () => {
    const { metadata } = parseFrontmatter(post)

    expect(metadata.title).toBe(
      'Suno Session Lab: turning clip generation into a navigable space',
    )
    expect(metadata.summary).toBe(
      'Turns music generation into a navigable space',
    )
    expect(metadata.repoUrl).toBe('https://github.com/NielsdaWheelz/suno-demo')
  })

  test('returns the body with the frontmatter block removed', () => {
    const { content } = parseFrontmatter(post)

    expect(content.startsWith('## why i built this')).toBe(true)
    expect(content).not.toContain('publishedAt')
  })

  // <time dateTime> and the newest-first sort both read this string directly;
  // a Date round-trip would shift it by a timezone.
  test('keeps publishedAt as the YYYY-MM-DD string it was written as', () => {
    expect(parseFrontmatter(post).metadata.publishedAt).toBe('2025-12-11')
  })

  test('throws when the frontmatter delimiters are missing', () => {
    expect(() => parseFrontmatter('# just a heading\n')).toThrow(/frontmatter/)
  })

  test('throws on a line that is not "key: value"', () => {
    expect(() => parseFrontmatter('---\ntitle\n---\n\nbody\n')).toThrow(
      /key: value/,
    )
  })
})
