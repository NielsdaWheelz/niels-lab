/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { mapEvents } from './github'

// Captured from https://api.github.com/users/NielsdaWheelz/events/public
// on 2026-08-09; whole events, only the actor block removed. The public
// events API sends no commit messages and no merged flag — a PushEvent
// carries ref + head sha, a merge arrives as action 'merged'.
const captured = [
  {
    id: '17178080950',
    type: 'PushEvent',
    repo: {
      id: 1114777227,
      name: 'NielsdaWheelz/nexus-web',
      url: 'https://api.github.com/repos/NielsdaWheelz/nexus-web',
    },
    payload: {
      repository_id: 1114777227,
      push_id: 39495412339,
      ref: 'refs/heads/main',
      head: '4fc8401fec9e8ba218d156a0664943303af57285',
      before: '9e3c3c48483297d3b41ed128b2bd1dd63ae92f36',
    },
    public: true,
    created_at: '2026-08-09T08:55:04Z',
  },
  {
    id: '17174140354',
    type: 'PushEvent',
    repo: {
      id: 1114777227,
      name: 'NielsdaWheelz/nexus-web',
      url: 'https://api.github.com/repos/NielsdaWheelz/nexus-web',
    },
    payload: {
      repository_id: 1114777227,
      push_id: 39491454360,
      ref: 'refs/heads/main',
      head: '9e3c3c48483297d3b41ed128b2bd1dd63ae92f36',
      before: 'd2d4c9a7aa9b8f96754049ff731ea36508b7d752',
    },
    public: true,
    created_at: '2026-08-09T07:49:42Z',
  },
  {
    id: '17172612764',
    type: 'PushEvent',
    repo: {
      id: 1114777227,
      name: 'NielsdaWheelz/nexus-web',
      url: 'https://api.github.com/repos/NielsdaWheelz/nexus-web',
    },
    payload: {
      repository_id: 1114777227,
      push_id: 39489918851,
      ref: 'refs/heads/codex/host-memory-reserve-envelope',
      head: '7d2f4f8c7106cce0a55707d6585e928f2764ba6a',
      before: '15039b872b9c62147aa9ead4ca1d361a6506dd26',
    },
    public: true,
    created_at: '2026-08-09T07:25:28Z',
  },
  {
    id: '17108997458',
    type: 'PushEvent',
    repo: {
      id: 1114777227,
      name: 'NielsdaWheelz/nexus-web',
      url: 'https://api.github.com/repos/NielsdaWheelz/nexus-web',
    },
    payload: {
      repository_id: 1114777227,
      push_id: 39426060060,
      ref: 'refs/heads/main',
      head: '0d12edf6e7f3b757c244104329649ed335f29393',
      before: '784f424d73d762b5eab8a76c1df81a378a9b5f8c',
    },
    public: true,
    created_at: '2026-08-08T06:59:00Z',
  },
  {
    id: '12996671919',
    type: 'PullRequestEvent',
    repo: {
      id: 1114777227,
      name: 'NielsdaWheelz/nexus-web',
      url: 'https://api.github.com/repos/NielsdaWheelz/nexus-web',
    },
    payload: {
      action: 'merged',
      number: 179,
      pull_request: {
        url: 'https://api.github.com/repos/NielsdaWheelz/nexus-web/pulls/179',
        id: 4237729484,
        number: 179,
        head: {
          ref: 'codex/converge-memory-swap',
          sha: '066a005ed56452a371f637d44efd6c0a4d0e6783',
        },
        base: {
          ref: 'main',
          sha: '9e3c3c48483297d3b41ed128b2bd1dd63ae92f36',
        },
      },
    },
    public: true,
    created_at: '2026-08-09T08:55:03Z',
  },
  {
    id: '12996256128',
    type: 'PullRequestEvent',
    repo: {
      id: 1114777227,
      name: 'NielsdaWheelz/nexus-web',
      url: 'https://api.github.com/repos/NielsdaWheelz/nexus-web',
    },
    payload: {
      action: 'opened',
      number: 179,
      pull_request: {
        url: 'https://api.github.com/repos/NielsdaWheelz/nexus-web/pulls/179',
        id: 4237729484,
        number: 179,
        head: {
          ref: 'codex/converge-memory-swap',
          sha: '066a005ed56452a371f637d44efd6c0a4d0e6783',
        },
        base: {
          ref: 'main',
          sha: '9e3c3c48483297d3b41ed128b2bd1dd63ae92f36',
        },
      },
    },
    public: true,
    created_at: '2026-08-09T08:33:10Z',
  },
  {
    id: '12993639757',
    type: 'PullRequestEvent',
    repo: {
      id: 1077887093,
      name: 'NielsdaWheelz/niels-lab',
      url: 'https://api.github.com/repos/NielsdaWheelz/niels-lab',
    },
    payload: {
      action: 'closed',
      number: 39,
      pull_request: {
        url: 'https://api.github.com/repos/NielsdaWheelz/niels-lab/pulls/39',
        id: 3087344666,
        number: 39,
        head: {
          ref: 'vercel/react-flightnextjs-rce-advisor-5vi9ad',
          sha: '330eb50aa1ea4fefccb062fb486b81e69bdd2424',
        },
        base: {
          ref: 'main',
          sha: '9cd0f3e8902682dab7a5cfa1a49be540914a06ea',
        },
      },
    },
    public: true,
    created_at: '2025-12-09T22:30:25Z',
  },
  {
    id: '17209190622',
    type: 'DeleteEvent',
    repo: {
      id: 1114777227,
      name: 'NielsdaWheelz/nexus-web',
      url: 'https://api.github.com/repos/NielsdaWheelz/nexus-web',
    },
    payload: {
      ref: 'codex/testing-standards-hard-cutover',
      ref_type: 'branch',
      full_ref: 'refs/heads/codex/testing-standards-hard-cutover',
      pusher_type: 'user',
    },
    public: true,
    created_at: '2026-08-09T16:14:25Z',
  },
  {
    id: '17176684762',
    type: 'CreateEvent',
    repo: {
      id: 1114777227,
      name: 'NielsdaWheelz/nexus-web',
      url: 'https://api.github.com/repos/NielsdaWheelz/nexus-web',
    },
    payload: {
      ref: 'codex/converge-memory-swap',
      ref_type: 'branch',
      full_ref: 'refs/heads/codex/converge-memory-swap',
      master_branch: 'main',
      description: null,
      pusher_type: 'user',
    },
    public: true,
    created_at: '2026-08-09T08:32:55Z',
  },
]

describe('mapEvents', () => {
  // One assertion holds the whole contract: a day's work on a repository is
  // one row, at the sha its trunk ended on. Side branches, pull requests and
  // branch churn are process, not shipping, and leave no trace.
  test('records the public event feed as one ship row per repository per day', () => {
    expect(mapEvents(captured)).toEqual([
      {
        date: '2026-08-09',
        kind: 'ship',
        text: 'nexus-web · main 4fc8401',
        href: 'https://github.com/NielsdaWheelz/nexus-web/commit/4fc8401fec9e8ba218d156a0664943303af57285',
      },
      {
        date: '2026-08-08',
        kind: 'ship',
        text: 'nexus-web · main 0d12edf',
        href: 'https://github.com/NielsdaWheelz/nexus-web/commit/0d12edf6e7f3b757c244104329649ed335f29393',
      },
    ])
  })

  test('records nothing for malformed or unexpected payloads', () => {
    expect(mapEvents({ message: 'Not Found' })).toEqual([])
    expect(mapEvents(null)).toEqual([])
    expect(
      mapEvents([
        null,
        'push',
        { type: 'PushEvent' },
        { type: 'PushEvent', repo: {}, payload: {}, created_at: '' },
      ]),
    ).toEqual([])
  })
})
