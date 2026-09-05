// The hour table (spec §6, §13 Q2 — client-approved 2026-09-05). Each band
// names a corpus entry by (slug, index) so the epigraph rendered on `/` is
// byte-identical to the list entry by construction, and which list opens
// by default. validate-content.mjs checks every ref resolves.

import type { HourBand } from '@/lib/hour'

export type HourTable = Record<
  HourBand,
  {
    epigraph: { list: string; index: number }
    opens: string
  }
>

export const hourTable: HourTable = {
  'small-hours': {
    epigraph: { list: 'things-that-are-distant-though-near', index: 6 },
    opens: 'things-that-are-distant-though-near',
  },
  dawn: {
    epigraph: { list: 'things-that-quicken-the-heart', index: 0 },
    opens: 'things-that-quicken-the-heart',
  },
  morning: {
    epigraph: { list: 'things-that-quicken-the-heart', index: 3 },
    opens: 'things-i-have-built',
  },
  afternoon: {
    epigraph: { list: 'things-that-should-be-small', index: 0 },
    opens: 'things-that-should-be-small',
  },
  dusk: {
    epigraph: { list: 'things-that-are-distant-though-near', index: 3 },
    opens: 'things-now-in-decline',
  },
  evening: {
    epigraph: { list: 'things-that-are-distant-though-near', index: 2 },
    opens: 'elegant-things',
  },
}
