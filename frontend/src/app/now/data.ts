export type NowEntry = {
  title: string
  detail: string
  href?: string
  linkLabel?: string
}

export type NowSection = {
  id: string
  index: string
  heading: string
  entries: NowEntry[]
}

// bump this whenever the page content changes — the margin note is derived
// from it, no Date.now() needed
export const updatedAt = '2026-07-20'

export const nowSections: NowSection[] = [
  {
    id: 'currently',
    index: '01 / currently',
    heading: 'Six-day weeks, on purpose.',
    entries: [
      {
        title: 'Fractal Tech',
        detail:
          'Software engineering fellow — six-day weeks, thirteen-hour days, shipping real systems instead of studying them.',
      },
    ],
  },
  {
    id: 'building',
    index: '02 / building',
    heading: 'Two systems, in parallel.',
    entries: [
      {
        title: 'Nexus',
        detail:
          'A reading, notes, and AI workspace that turns highlights, notes, and chat into one shared resource graph, so a citation stays a live link back to its evidence instead of a dead excerpt.',
        href: 'https://nexus.nielseriknandal.com',
        linkLabel: 'nexus.nielseriknandal.com',
      },
      {
        title: 'niels-gpt',
        detail:
          'A tiny, complete LLM system — tokenizer, cache, pretrain, SFT, kv-cache inference, chat CLI — built laptop-first with strict guardrails instead of borrowed magic.',
        href: 'https://github.com/NielsdaWheelz/niels-gpt',
        linkLabel: 'source on GitHub',
      },
    ],
  },
  {
    id: 'open-to',
    index: '03 / open to',
    heading: 'The next hard problem.',
    entries: [
      {
        title: 'Full-time AI / software engineering roles',
        detail:
          'Model internals, deterministic backends, or the interfaces that connect them — that range is where I do my best work.',
      },
    ],
  },
  {
    id: 'this-site',
    index: '04 / on this site',
    heading: 'Never quite finished.',
    entries: [
      {
        title: 'Dark mode',
        detail: 'Landed recently, so this page reads fine at 1am too.',
      },
      {
        title: 'ctrl+k terminal',
        detail:
          'A keyboard launcher for getting around the site without reaching for the mouse.',
      },
      {
        title: '/lab',
        detail:
          'Interactive model-internals experiments, starting with a token-sampling playground — temperature, top-k, and top-p you can actually feel.',
        href: '/lab',
        linkLabel: 'open the lab',
      },
    ],
  },
]
