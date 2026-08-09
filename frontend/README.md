Frontend app for `niels-lab` — The Pillow Book.

## Install

- `bun install`

## Run

- `bun run dev`
- `bun run check`
- `bun run build`
- `bun run validate:content`

## What is here

- `/` — the book's index: eight lists, entry counts, dates.
- `/lists/[slug]` — one list per page, evidence inline as `<details>`.
- `/log` — the ledger: hand-written rows merged with GitHub activity (ISR, daily).
- `/writing`, `/projects` — essays and working notes from MDX.
- `/cv`, `/now`, `/colophon`, `/lab` — record, diary, method, playground.
- `/rss`, `/sitemap.xml`, `/llms.txt`, `/llms-full.txt` — generated from the
  same typed content the pages render.

Content lives in `src/content/lists.ts`, `src/content/log.ts`,
`src/app/{writing,projects}/posts/*.mdx`, and `src/app/cv/data.ts`. The lists
are TypeScript, not markdown: the type checker is the CMS.

Two client components run: the theme toggle and the lab's sampling playground.
Everything else is server-rendered markup, and every load-bearing sentence is in
the initial HTML.

## Deploy

Vercel Root Directory must be `frontend`.

Set `NEXT_PUBLIC_SITE_URL` if you want to override the canonical site URL outside production.

Enable Vercel system environment variables if you want preview and production host detection without hardcoding hosts in the app.

## Verification

- formatting: `bun run format:check`
- lint: `bun run lint`
- types: `bun run typecheck`
- tests: `bun run test`
- content: `bun run validate:content` (also HEAD-checks every evidence link)
- build: `bun run build`
