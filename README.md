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
- `frontend/src/app/cv/cv.mdx`

Canonical routes are `/`, `/projects`, `/writing`, and `/cv`.

The terminal is global and opens with `ctrl+k`.

The validator matches the current loaders: frontmatter uses `---` delimiters, one `key: value` entry per line, and `publishedAt` uses `YYYY-MM-DD`.

## Deploy

Vercel should point its Root Directory at `frontend`.

Set `NEXT_PUBLIC_NIELS_GPT_API_BASE` for terminal chat. If you want to override the canonical site URL locally, set `NEXT_PUBLIC_SITE_URL`.

If you want preview deployments to derive the correct production and preview hosts automatically, enable Vercel's "Automatically expose System Environment Variables" setting.
