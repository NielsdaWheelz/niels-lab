# niels-lab

Frontend-only portfolio app.

This repo is Bun-first.

## Run

- `cd frontend && bun install`
- `cd frontend && bun run dev`
- `cd frontend && bun run check`

Canonical frontend content is:

- `frontend/src/app/projects/posts/*.mdx`
- `frontend/src/app/writing/posts/*.mdx`
- `frontend/src/app/cv/data.ts`

Canonical routes are `/`, `/projects`, `/writing`, and `/cv`.

Machine-readable discovery surfaces are `/robots.txt`, `/sitemap.xml`, `/rss`,
`/llms.txt`, and `/llms-full.txt`. Their canonical production domain is
`https://nielseriknandal.com`.

The site has light ("day paper") and dark ("midnight blueprint") themes: toggled in the nav. Theme state lives on `html[data-theme]` with the engine in `frontend/src/lib/theme.ts`.

The validator matches the current loaders: frontmatter uses `---` delimiters, one `key: value` entry per line, and `publishedAt` uses `YYYY-MM-DD`.

## Deploy

Vercel should point its Root Directory at `frontend`.

If you want to override the canonical site URL locally, set `NEXT_PUBLIC_SITE_URL`.

If you want preview deployments to derive the correct production and preview hosts automatically, enable Vercel's "Automatically expose System Environment Variables" setting.
