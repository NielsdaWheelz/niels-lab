// The Pillow Book corpus. Typed data is the CMS (docs/pillow-book-spec.md §5):
// entries are ordered by the author, never sorted; evidence is primary proof or absent.

export type Evidence =
  | { label: string; href: string } // primary proof: commit, video, meet result, post
  | { label: string; src: string; kind: 'image' } // embedded asset in /public/evidence/

export type Entry = {
  text: string // 5–25 words, one image or one judgment
  evidence?: Evidence // omit when no proof exists; never placeholder
  added: `${number}-${number}-${number}` // YYYY-MM-DD
}

export type PillowList = {
  slug: string // kebab-case, stable, used in URL + anchors
  title: string // the merciless heading
  note?: string // optional one-line gloss under the heading
  entries: Entry[] // ordered by the author, not by date
}

// Derived (never stored): count = entries.length; lastWritten = max(added).

export const lists: PillowList[] = [
  {
    slug: 'things-that-quicken-the-heart',
    title: 'Things that quicken the heart',
    entries: [
      {
        text: "The bar's first millimeter off the floor, which tells you everything the next three seconds will confirm.",
        added: '2026-08-09',
      },
      {
        text: "Watching a meet total pass 1,000 on a third attempt you almost didn't take.",
        added: '2026-08-09',
      },
      {
        text: 'A keeper off his line while the ball is still rolling toward your foot.',
        added: '2026-08-09',
      },
      {
        text: 'Levin mowing, at the moment the scythe begins to swing itself.',
        added: '2026-08-09',
      },
      {
        text: 'The tokenizer round-tripping a page of Joyce with zero unknowns. Byte fallback, keeping its promise.',
        evidence: {
          label: 'niels-gpt tokenizer, byte fallback',
          href: 'https://github.com/NielsdaWheelz/niels-gpt-2',
        },
        added: '2026-08-09',
      },
      {
        text: 'Two generation paths, one fast, one naive, printing identical tokens. The speedup is now legal.',
        evidence: {
          label: 'niels-gpt: kv-cache prefill/decode against naive generation',
          href: '/projects/niels-gpt',
        },
        added: '2026-08-09',
      },
      {
        text: "A stranger's heart rate slowing on schedule, inside a game you built to slow it.",
        added: '2026-08-09',
      },
    ],
  },
  {
    slug: 'things-i-have-built',
    title: 'Things I have built',
    note: 'The résumé. Every line links to the thing itself.',
    entries: [
      {
        text: 'A small language model, by hand, in two days — tokenizer, cache, pretraining, chat. It hedges where I hedge.',
        evidence: {
          label: 'niels-gpt, working notes',
          href: '/projects/niels-gpt',
        },
        added: '2026-08-09',
      },
      {
        text: 'A reading system where a highlight is a byte offset into canonical text, so citations survive the document changing beneath them.',
        evidence: {
          label: 'Nexus, source',
          href: 'https://github.com/NielsdaWheelz/nexus-web',
        },
        added: '2026-08-09',
      },
      {
        text: 'LLM agents wrapped around a deterministic scheduler. They may explain the plan; they may not touch it.',
        evidence: {
          label: 'Factory Simulator, working notes',
          href: '/projects/factory-simulator',
        },
        added: '2026-08-09',
      },
      {
        text: 'A reader that ingests messy PDFs and answers from the text it was given, not the text it wished for.',
        evidence: {
          label: 'Fractal Chat, working notes',
          href: '/projects/fractal-chat',
        },
        added: '2026-08-09',
      },
      {
        text: 'A real-time multiplayer Go board, built for a friend, who thanked me by winning on it.',
        evidence: {
          label: 'Fractal Go, working notes',
          href: '/projects/fractal-go',
        },
        added: '2026-08-09',
      },
      {
        text: "A map of a music model's range: clips clustered by sound, regeneration steered toward the region you point at.",
        evidence: {
          label: 'Suno Session Lab, working notes',
          href: '/projects/suno-demo',
        },
        added: '2026-08-09',
      },
      {
        text: 'Two plain-English walkthroughs of backpropagation, written for the person I was in October.',
        evidence: {
          label: 'Neural networks from zero',
          href: '/writing/zero-to-hero-one',
        },
        added: '2026-08-09',
      },
      {
        text: 'This site: typed data for a CMS, evidence links checked at build, one client component. The colophon confesses.',
        evidence: {
          label: 'niels-lab, source',
          href: 'https://github.com/NielsdaWheelz/niels-lab',
        },
        added: '2026-08-09',
      },
    ],
  },
  {
    slug: 'hateful-things',
    title: 'Hateful things (found in codebases)',
    entries: [
      {
        text: 'A mock of the unit under test. The suite is green because it agrees with itself.',
        added: '2026-08-09',
      },
      {
        text: "A catch block holding a comment where the plan should be. 'This shouldn't happen.' It is happening now.",
        added: '2026-08-09',
      },
      {
        text: 'A token cache with no record of which tokenizer wrote it. Every downstream loss curve is a rumor.',
        evidence: {
          label: 'niels-gpt caches hard-fail on tokenizer SHA mismatch',
          href: 'https://huggingface.co/nnandal/niels-gpt',
        },
        added: '2026-08-09',
      },
      {
        text: "Sentinel tokens spelled as plain strings. 'Hope no one types <|system|>' is not a strategy.",
        evidence: {
          label: 'niels-gpt sentinel-token collision checks',
          href: 'https://github.com/NielsdaWheelz/niels-gpt-2',
        },
        added: '2026-08-09',
      },
      {
        text: 'Annotations pinned to line numbers. One re-ingestion later, every highlight decorates the wrong sentence.',
        evidence: {
          label: 'Nexus stores byte-offset ranges, remapped on re-ingestion',
          href: '/projects/nexus',
        },
        added: '2026-08-09',
      },
      {
        text: "A spotter's hands on a bar that was still moving. Nobody will ever know what the rep was worth.",
        added: '2026-08-09',
      },
      {
        text: 'An epigraph from The Art of War at the head of a CRUD service.',
        added: '2026-08-09',
      },
      {
        text: 'A schema change applied by hand in psql, remembered by no one, discovered by the next deploy.',
        added: '2026-08-09',
      },
    ],
  },
  {
    slug: 'elegant-things',
    title: 'Elegant things',
    entries: [
      {
        text: 'Squat bar path: one vertical line, drawn twice.',
        added: '2026-08-09',
      },
      {
        text: 'A saucer pass, airborne over one stick and flat on arrival.',
        added: '2026-08-09',
      },
      {
        text: 'The first sentence of Pride and Prejudice: a thesis, a joke, and the entire plot.',
        added: '2026-08-09',
      },
      {
        text: "Dickinson's slant rhyme: the bell you expected, struck slightly off, on purpose.",
        added: '2026-08-09',
      },
      {
        text: 'Braudel: the sea gets the biography, and kings are demoted to weather.',
        added: '2026-08-09',
      },
      {
        text: 'RMSNorm: LayerNorm minus the mean, and nothing misses it.',
        evidence: {
          label: 'niels-gpt model code',
          href: 'https://github.com/NielsdaWheelz/niels-gpt-2',
        },
        added: '2026-08-09',
      },
      {
        text: 'Weight tying: the matrix that reads is the matrix that speaks.',
        evidence: {
          label: 'niels-gpt checkpoint',
          href: 'https://huggingface.co/nnandal/niels-gpt',
        },
        added: '2026-08-09',
      },
      {
        text: 'A citation that still resolves after the document was edited underneath it.',
        evidence: {
          label: 'Nexus, live',
          href: 'https://nexus.nielseriknandal.com',
        },
        added: '2026-08-09',
      },
    ],
  },
  {
    slug: 'things-that-should-be-small',
    title: 'Things that should be small',
    entries: [
      {
        text: 'The blast radius of a deploy. If you cannot draw it, it is not small.',
        added: '2026-08-09',
      },
      {
        text: 'A pull request. Past the second scroll, the review becomes a skim and the approval becomes a signature.',
        added: '2026-08-09',
      },
      {
        text: 'The distance between a claim and its proof. Ideally one link, to a trace.',
        evidence: {
          label: 'Factory Simulator ships a full pipeline trace per run',
          href: 'https://github.com/NielsdaWheelz/factory-simulator',
        },
        added: '2026-08-09',
      },
      {
        text: 'The gap between two touches of the ball. Possession is overhead.',
        added: '2026-08-09',
      },
      {
        text: 'The walkout before a heavy squat. Two steps. Every extra step is paid for at the bottom.',
        added: '2026-08-09',
      },
      {
        text: "An agent's opportunity to improvise. Pydantic contract in, Pydantic contract out, nothing invented in between.",
        evidence: {
          label: 'Factory Simulator contracts',
          href: '/projects/factory-simulator',
        },
        added: '2026-08-09',
      },
      {
        text: 'A migration. One concern per revision, so the rollback is a sentence instead of a meeting.',
        added: '2026-08-09',
      },
      {
        text: 'A Dickinson stanza. Four lines, and you are not the person who started reading them.',
        added: '2026-08-09',
      },
    ],
  },
  {
    slug: 'rare-things',
    title: 'Rare things',
    entries: [
      {
        text: 'A powerlifting meet that runs on schedule.',
        added: '2026-08-09',
      },
      {
        text: 'A benched player who agrees with the decision.',
        added: '2026-08-09',
      },
      {
        text: 'Someone who finishes Durant: eleven volumes, in order, no skimming the Reformation.',
        added: '2026-08-09',
      },
      {
        text: 'An analysis that matches its preregistration.',
        added: '2026-08-09',
      },
      {
        text: 'A rewrite that is smaller than the thing it replaced.',
        added: '2026-08-09',
      },
      {
        text: 'A model card that admits what the model cannot do.',
        evidence: {
          label: 'one attempt',
          href: 'https://huggingface.co/nnandal/niels-gpt',
        },
        added: '2026-08-09',
      },
      {
        text: 'A secondhand bookshop that has shelved Spengler under Self-Help.',
        added: '2026-08-09',
      },
    ],
  },
  {
    slug: 'things-that-are-distant-though-near',
    title: 'Things that are distant though near',
    entries: [
      {
        text: 'The last two inches of a deadlift that has stopped moving.',
        added: '2026-08-09',
      },
      {
        text: 'The thousand-pound club, from nine ninety-five.',
        added: '2026-08-09',
      },
      {
        text: 'War and Peace, third shelf, eye level. I have meant to reread it for ten years and can touch it from this chair.',
        added: '2026-08-09',
      },
      {
        text: 'The net, in a shootout. Four feet by six, receding as you skate in.',
        added: '2026-08-09',
      },
      {
        text: 'The end of Ulysses, from anywhere in Ulysses.',
        added: '2026-08-09',
      },
      { text: 'Statistical significance, at p = 0.051.', added: '2026-08-09' },
      { text: 'Sleep, with the diff still open.', added: '2026-08-09' },
    ],
  },
  {
    slug: 'things-now-in-decline',
    title: 'Things now in decline',
    entries: [
      {
        text: 'The two-hour practice nobody films. A proper bag skate produces no content.',
        added: '2026-08-09',
      },
      {
        text: 'Defending, in pickup soccer. Eleven wingers a side now.',
        added: '2026-08-09',
      },
      { text: 'Commit messages that explain why.', added: '2026-08-09' },
      {
        text: 'The personal website. This one included; it knows what it is.',
        evidence: { label: 'the colophon', href: '/colophon' },
        added: '2026-08-09',
      },
      {
        text: 'The incentivized choice experiment. Cheaper now to ask a model what humans would want.',
        added: '2026-08-09',
      },
      {
        text: 'Carrying poems in memory. I keep two Keats odes there against the general trend.',
        added: '2026-08-09',
      },
      { text: 'Hotel gyms with an actual barbell.', added: '2026-08-09' },
    ],
  },
]

// YYYY-MM-DD compares lexicographically, so string max is date max.
// Invariant: a PillowList has at least one entry.
export function lastWritten(list: PillowList): Entry['added'] {
  return list.entries
    .map((entry) => entry.added)
    .reduce((latest, added) => (added > latest ? added : latest))
}
