import type { LedgerEvent } from '@/lib/ledger/types'

// Hand-written ledger rows. Every href is a verified primary source; a row
// without proof ships without one.
export const log: LedgerEvent[] = [
  { date: '2026-06-05', kind: 'train', text: 'SQ 5×5 @335 · pickup 90 min' },
  {
    date: '2026-06-12',
    kind: 'read',
    text: 'Braudel 41pp · grain prices, Messina, the slowness of news',
  },
  {
    date: '2026-06-19',
    kind: 'train',
    text: 'DL 3×3 @315 · hook grip holding',
  },
  {
    date: '2026-06-26',
    kind: 'write',
    text: 'list drafts · 31 entries, 9 survived the evening read',
  },
  {
    date: '2026-07-03',
    kind: 'read',
    text: 'Moby-Dick 40pp before work · cetology, skimmed nothing',
  },
  {
    date: '2026-07-10',
    kind: 'train',
    text: 'BP 3×3 @250 · paused, no bounce',
  },
  {
    date: '2026-07-20',
    kind: 'ship',
    text: '/og 500 in production · fonts missing from the function bundle',
    failed: {
      lesson: 'A build that passes locally proves the laptop, not the deploy.',
    },
    href: 'https://github.com/NielsdaWheelz/niels-lab/commit/d659e81273db360b5ba911607a0a292fcb97e023',
  },
  {
    date: '2026-07-20',
    kind: 'ship',
    text: '/lab, /now, /colophon, llms.txt, print CV · the notebook opens up',
    href: 'https://github.com/NielsdaWheelz/niels-lab/commit/03d5b449999449bbd33d51b331dbe174fff0a940',
  },
  {
    date: '2026-08-02',
    kind: 'train',
    text: 'SQ 3×3 @385 · third set moved like the first',
  },
  {
    date: '2026-08-08',
    kind: 'write',
    text: 'pillow book corpus · ~140 candidates cut to 60',
  },
  {
    date: '2026-08-09',
    kind: 'ship',
    text: 'spec and rules codified · docs/pillow-book-spec.md, docs/rules.md',
  },
]
