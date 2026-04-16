# niels-lab

Frontend-only portfolio app.

## Run

- `cd frontend && npm run dev`
- `cd frontend && npm run build`
- `cd frontend && npm run validate:content`

Canonical frontend content is:

- `frontend/src/app/projects/posts/*.mdx`
- `frontend/src/app/writing/posts/*.mdx`
- `frontend/src/app/cv/cv.mdx`

Canonical routes are `/`, `/projects`, `/writing`, and `/cv`.

The terminal is global and opens with `ctrl+k`.

The validator matches the current loaders: frontmatter uses `---` delimiters, one `key: value` entry per line, and `publishedAt` uses `YYYY-MM-DD`.
