# niels-lab

Frontend-only site: The Pillow Book, a zuihitsu of dated lists in which every
checkable claim links to its evidence.

This repo is Bun-first.

## Run

- `cd frontend && bun install`
- `cd frontend && bun run dev`
- `cd frontend && bun run check`

Canonical frontend content is:

- `frontend/src/content/lists.ts` (the lists — typed data, reviewed like code)
- `frontend/src/content/log.ts` (hand-written ledger rows)
- `frontend/src/app/projects/posts/*.mdx`
- `frontend/src/app/writing/posts/*.mdx`
- `frontend/src/app/cv/data.ts`

Canonical routes are `/`, `/lists/[slug]`, `/log`, `/projects`, `/writing`,
`/cv`, `/now`, `/colophon`, and `/lab`.

Machine-readable discovery surfaces are `/robots.txt`, `/sitemap.xml`, `/rss`,
`/llms.txt`, and `/llms-full.txt`. They are generated from the same typed
content the pages render, so they cannot drift. Their canonical production
domain is `https://nielseriknandal.com`. Nothing in `public/` may share a path
with a route: Next.js serves a 500 for the collision rather than picking a
winner.

Dark is the canonical theme, light its inverse; toggled in the nav. Theme state
lives on `html[data-theme]` with the engine in `frontend/src/lib/theme.ts`.

The validator matches the current loaders: frontmatter uses `---` delimiters,
one `key: value` entry per line, and `publishedAt` uses `YYYY-MM-DD`. It also
gates the lists — slug shape, entry length, dates, evidence shape, and the
liveness of every evidence link.

## Deploy

Vercel should point its Root Directory at `frontend`.

If you want to override the canonical site URL locally, set `NEXT_PUBLIC_SITE_URL`.

If you want preview deployments to derive the correct production and preview hosts automatically, enable Vercel's "Automatically expose System Environment Variables" setting.
