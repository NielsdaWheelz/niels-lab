Frontend app for `niels-lab`.

## Install

- `bun install`

## Run

- `bun run dev`
- `bun run check`
- `bun run build`
- `bun run validate:content`

## Deploy

Vercel Root Directory must be `frontend`.

Set `NEXT_PUBLIC_NIELS_GPT_API_BASE` in Vercel and local `.env.local` if you want terminal chat enabled.

Set `NEXT_PUBLIC_SITE_URL` if you want to override the canonical site URL outside production.

Enable Vercel system environment variables if you want preview and production host detection without hardcoding hosts in the app.

## Verification

- formatting: `bun run format:check`
- lint: `bun run lint`
- types: `bun run typecheck`
- content: `bun run validate:content`
- build: `bun run build`
