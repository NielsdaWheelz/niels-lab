#!/usr/bin/env node

import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const frontendRoot = path.resolve(__dirname, '..')
const frontmatterRegex = /---\s*([\s\S]*?)\s*---/
const publishedAtRegex = /^\d{4}-\d{2}-\d{2}$/

const listsFile = 'src/content/lists.ts'
const logFile = 'src/content/log.ts'
const hoursFile = 'src/content/hours.ts'
const inkAgeFile = 'src/lib/ledger/inkAge.ts'
const logKinds = new Set(['train', 'read', 'ship', 'write'])
// Spec §6: America/Los_Angeles wall-clock bands, 00–05 · 05–08 · 08–12 ·
// 12–17 · 17–20 · 20–24.
const hourBands = [
  'small-hours',
  'dawn',
  'morning',
  'afternoon',
  'dusk',
  'evening',
]
const slugRegex = /^[a-z0-9]+(?:-[a-z0-9]+)*$/
// Lifting and meet numbers ship as prose only (spec §9): no evidence, ever.
// A keyword tripwire, not a complete gate: it catches the obvious nouns,
// not every phrasing of a lifting claim.
const liftingRegex = /\b(squat|bench|deadlift|total|lbs?|kg)\b/i
const linkTimeoutMs = 15000
const userAgent = 'nielseriknandal.com content validator'
const today = new Date().toISOString().slice(0, 10)

const contentSources = [
  {
    name: 'projects',
    directory: 'src/app/projects/posts',
    requiredKeys: ['title', 'publishedAt', 'summary'],
  },
  {
    name: 'writing',
    directory: 'src/app/writing/posts',
    requiredKeys: ['title', 'publishedAt', 'summary'],
  },
]

function stripOuterQuotes(value) {
  return value.replace(/^['"](.*)['"]$/, '$1')
}

function parseFrontmatter(rawContent, relativePath, errors) {
  const match = frontmatterRegex.exec(rawContent)

  if (!match) {
    errors.push(`${relativePath}: missing frontmatter delimiters (---)`)
    return null
  }

  const frontmatterBlock = match[1]
  const content = rawContent.replace(frontmatterRegex, '').trim()

  if (!frontmatterBlock.trim()) {
    errors.push(`${relativePath}: frontmatter is empty`)
    return null
  }

  const metadata = {}
  const lines = frontmatterBlock.trim().split(/\r?\n/)

  for (const [index, line] of lines.entries()) {
    if (!line.trim()) {
      errors.push(
        `${relativePath}:${index + 1}: blank lines inside frontmatter are not supported`,
      )
      continue
    }

    if (!line.includes(': ')) {
      errors.push(
        `${relativePath}:${index + 1}: frontmatter lines must use "key: value"`,
      )
      continue
    }

    const [rawKey, ...rawValueParts] = line.split(': ')
    const key = rawKey.trim()

    if (!key) {
      errors.push(`${relativePath}:${index + 1}: frontmatter key is empty`)
      continue
    }

    if (Object.prototype.hasOwnProperty.call(metadata, key)) {
      errors.push(
        `${relativePath}:${index + 1}: duplicate frontmatter key "${key}"`,
      )
      continue
    }

    metadata[key] = stripOuterQuotes(rawValueParts.join(': ').trim())
  }

  return { metadata, content }
}

function validatePublishedAt(value, relativePath, errors) {
  if (!publishedAtRegex.test(value)) {
    errors.push(`${relativePath}: publishedAt must use YYYY-MM-DD`)
    return
  }

  const parsed = new Date(`${value}T00:00:00Z`)
  if (Number.isNaN(parsed.valueOf())) {
    errors.push(`${relativePath}: publishedAt is not a valid date`)
  }
}

function validateUrl(value, fieldName, relativePath, errors) {
  try {
    new URL(value)
  } catch {
    errors.push(`${relativePath}: ${fieldName} must be an absolute URL`)
  }
}

function validateMetadata(metadata, content, source, relativePath, errors) {
  if (typeof metadata.title !== 'string' || metadata.title.trim() === '') {
    errors.push(`${relativePath}: title must be present and non-empty`)
  }

  for (const fieldName of ['repoUrl', 'liveUrl']) {
    if (
      typeof metadata[fieldName] === 'string' &&
      metadata[fieldName].trim() !== ''
    ) {
      validateUrl(metadata[fieldName], fieldName, relativePath, errors)
    }
  }

  // A draft needs only title + draft; its body may be empty and its
  // publishedAt is checked only when present. Deleting the draft key is the
  // act of publishing — everything below is what publishing requires.
  if (metadata.draft !== undefined) {
    if (metadata.draft !== 'true') {
      errors.push(`${relativePath}: draft, when present, must be exactly true`)
    }

    if (metadata.publishedAt !== undefined) {
      validatePublishedAt(metadata.publishedAt, relativePath, errors)
    }

    return
  }

  for (const key of source.requiredKeys) {
    if (!Object.prototype.hasOwnProperty.call(metadata, key)) {
      errors.push(`${relativePath}: missing required frontmatter key "${key}"`)
    }
  }

  if (
    typeof metadata.publishedAt !== 'string' ||
    metadata.publishedAt.trim() === ''
  ) {
    errors.push(`${relativePath}: publishedAt must be present and non-empty`)
  } else {
    validatePublishedAt(metadata.publishedAt, relativePath, errors)
  }

  if (typeof metadata.summary !== 'string' || metadata.summary.trim() === '') {
    errors.push(`${relativePath}: summary must be present and non-empty`)
  }

  if (!content) {
    errors.push(`${relativePath}: content body is empty`)
  }
}

function getSourceFiles(source, errors) {
  const absoluteDirectory = path.join(frontendRoot, source.directory)
  if (!fs.existsSync(absoluteDirectory)) {
    errors.push(`${source.directory}: content directory is missing`)
    return []
  }

  const files = fs
    .readdirSync(absoluteDirectory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && path.extname(entry.name) === '.mdx')
    .map((entry) => path.join(source.directory, entry.name))
    .sort()

  if (files.length === 0) {
    errors.push(`${source.directory}: no .mdx files found`)
  }

  return files
}

// The corpus is typed TypeScript, so bun — the repo's runner, and this
// script's parent process — reads the module and hands back plain JSON.
// Nothing here parses source text.
function loadExport(relativeFile, exportName, errors) {
  const modulePath = JSON.stringify(path.join(frontendRoot, relativeFile))
  const result = spawnSync(
    'bun',
    [
      '--eval',
      `process.stdout.write(JSON.stringify((await import(${modulePath})).${exportName} ?? null))`,
    ],
    { encoding: 'utf-8' },
  )

  if (result.status !== 0) {
    const reason = (result.stderr || result.error?.message || '').trim()
    errors.push(`${relativeFile}: could not be loaded (${reason})`)
    return []
  }

  const value = JSON.parse(result.stdout)
  if (value === null) {
    errors.push(`${relativeFile}: no "${exportName}" export`)
    return []
  }

  return value
}

function routeExists(pathname) {
  const directory = pathname === '/' ? '' : pathname

  return fs.existsSync(
    path.join(frontendRoot, 'src/app', directory, 'page.tsx'),
  )
}

function internalHrefResolves(href, listSlugs) {
  const pathname = href.split(/[?#]/)[0].replace(/(.)\/$/, '$1')
  const segments = pathname === '/' ? [] : pathname.slice(1).split('/')

  if (segments.length === 2) {
    const [section, slug] = segments

    if (section === 'writing' || section === 'projects') {
      const file = path.join(
        frontendRoot,
        'src/app',
        section,
        'posts',
        `${slug}.mdx`,
      )

      if (!fs.existsSync(file)) {
        return false
      }

      // A draft has no page: evidence pointing at one is dead proof.
      const match = frontmatterRegex.exec(fs.readFileSync(file, 'utf-8'))
      return match !== null && !/^draft:/m.test(match[1])
    }

    if (section === 'lists') {
      return listSlugs.has(slug)
    }
  }

  return (
    routeExists(pathname) ||
    fs.existsSync(path.join(frontendRoot, 'public', pathname))
  )
}

function validateEvidence(evidence, where, errors, listSlugs, externalHrefs) {
  if (typeof evidence !== 'object' || evidence === null) {
    errors.push(`${where}: evidence must be an object`)
    return
  }

  if (typeof evidence.label !== 'string' || evidence.label.trim() === '') {
    errors.push(`${where}: evidence.label must be present and non-empty`)
  }

  if (evidence.href !== undefined) {
    if (typeof evidence.href !== 'string' || evidence.href.trim() === '') {
      errors.push(`${where}: evidence.href must be a non-empty string`)
      return
    }

    if (evidence.href.startsWith('/')) {
      if (!internalHrefResolves(evidence.href, listSlugs)) {
        errors.push(
          `${where}: evidence.href "${evidence.href}" resolves to no route or file`,
        )
      }
      return
    }

    if (!/^https?:\/\//.test(evidence.href)) {
      errors.push(
        `${where}: evidence.href must be root-relative or an absolute http(s) URL`,
      )
      return
    }

    externalHrefs.add(evidence.href)
    return
  }

  if (evidence.kind !== 'image' || typeof evidence.src !== 'string') {
    errors.push(
      `${where}: evidence must carry either href, or src with kind "image"`,
    )
    return
  }

  validateImagePath(evidence.src, where, errors)
}

function validateEntry(entry, where, errors, listSlugs, externalHrefs) {
  if (typeof entry.text !== 'string' || entry.text.trim() === '') {
    errors.push(`${where}: text must be present and non-empty`)
  } else {
    const words = entry.text.trim().split(/\s+/).length

    if (words < 5 || words > 25) {
      errors.push(`${where}: text must be 5–25 words (found ${words})`)
    }

    if (entry.evidence !== undefined && liftingRegex.test(entry.text)) {
      errors.push(
        `${where}: lifting claims ship as prose, never evidence-marked`,
      )
    }
  }

  if (typeof entry.added !== 'string' || !publishedAtRegex.test(entry.added)) {
    errors.push(`${where}: added must use YYYY-MM-DD`)
  } else if (Number.isNaN(new Date(`${entry.added}T00:00:00Z`).valueOf())) {
    errors.push(`${where}: added is not a valid date`)
  } else if (entry.added > today) {
    errors.push(`${where}: added is in the future`)
  }

  if (entry.evidence !== undefined) {
    validateEvidence(entry.evidence, where, errors, listSlugs, externalHrefs)
  }
}

function validateLists(lists, errors, externalHrefs) {
  const listSlugs = new Set(lists.map((list) => list.slug))
  const seen = new Set()

  for (const list of lists) {
    const where = `${listsFile}: ${list.slug}`

    if (typeof list.slug !== 'string' || !slugRegex.test(list.slug)) {
      errors.push(`${listsFile}: slug "${list.slug}" must be kebab-case`)
    }

    if (seen.has(list.slug)) {
      errors.push(`${listsFile}: duplicate slug "${list.slug}"`)
    }
    seen.add(list.slug)

    if (typeof list.title !== 'string' || list.title.trim() === '') {
      errors.push(`${where}: title must be present and non-empty`)
    }

    if (!Array.isArray(list.entries) || list.entries.length === 0) {
      errors.push(`${where}: entries must be a non-empty array`)
      continue
    }

    for (const [index, entry] of list.entries.entries()) {
      validateEntry(
        entry,
        `${where} entry ${index + 1}`,
        errors,
        listSlugs,
        externalHrefs,
      )
    }
  }

  return lists.reduce((count, list) => count + list.entries.length, 0)
}

// The ledger's manual rows: mergeEvents sorts on date, isStale parses it,
// and every href is a claim of proof — all gated here, like the lists.
function validateLog(events, errors, externalHrefs) {
  if (!Array.isArray(events)) {
    errors.push(`${logFile}: log must be an array`)
    return
  }

  for (const [index, event] of events.entries()) {
    const where = `${logFile}: row ${index + 1}`

    if (typeof event.date !== 'string' || !publishedAtRegex.test(event.date)) {
      errors.push(`${where}: date must use YYYY-MM-DD`)
    } else if (Number.isNaN(new Date(`${event.date}T00:00:00Z`).valueOf())) {
      errors.push(`${where}: date is not a valid date`)
    } else if (event.date > today) {
      errors.push(`${where}: date is in the future`)
    }

    if (!logKinds.has(event.kind)) {
      errors.push(`${where}: kind must be one of train, read, ship, write`)
    }

    if (typeof event.text !== 'string' || event.text.trim() === '') {
      errors.push(`${where}: text must be present and non-empty`)
    }

    if (
      event.failed !== undefined &&
      typeof event.failed?.lesson !== 'string'
    ) {
      errors.push(`${where}: failed must carry a lesson`)
    }

    if (event.href !== undefined) {
      if (!/^https?:\/\//.test(event.href)) {
        errors.push(`${where}: href must be an absolute http(s) URL`)
      } else {
        externalHrefs.add(event.href)
      }
    }
  }
}

// Spec §6/§11: every band's epigraph must name a published list and an
// in-range entry index, so the rendered line is byte-identical to the
// corpus by construction; `opens` must resolve the same way.
function validateHours(hourTable, errors, lists) {
  const listBySlug = new Map(lists.map((list) => [list.slug, list]))

  for (const bandName of hourBands) {
    const entry = hourTable?.[bandName]
    const where = `${hoursFile}: band "${bandName}"`

    if (!entry) {
      errors.push(`${where} is missing`)
      continue
    }

    const epigraphList = listBySlug.get(entry.epigraph?.list)
    if (!epigraphList) {
      errors.push(
        `${where}: epigraph list "${entry.epigraph?.list}" is not a published list`,
      )
    } else if (
      !Number.isInteger(entry.epigraph?.index) ||
      entry.epigraph.index < 0 ||
      entry.epigraph.index >= epigraphList.entries.length
    ) {
      errors.push(
        `${where}: epigraph index ${entry.epigraph?.index} is out of range for "${entry.epigraph?.list}"`,
      )
    }

    if (!listBySlug.has(entry.opens)) {
      errors.push(`${where}: opens "${entry.opens}" is not a published list`)
    }
  }
}

// Spec §6: AGE_STEPS/FAILURE_FACTOR must stay finite numbers. Reuses
// loadExport rather than new machinery; ageStep itself isn't data and isn't
// checked here.
function validateInkAge(errors) {
  const ageSteps = loadExport(inkAgeFile, 'AGE_STEPS', errors)
  const failureFactor = loadExport(inkAgeFile, 'FAILURE_FACTOR', errors)
  const constants = [
    ...(Array.isArray(ageSteps) ? ageSteps : []),
    failureFactor,
  ]

  for (const value of constants) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      errors.push(`${inkAgeFile}: ink-age constants must be finite numbers`)
    }
  }
}

// Dead proof is a build error; an unreachable network is not, so the site
// still builds on a train.
async function checkExternalHrefs(hrefs, errors, warnings) {
  const request = (href, method) =>
    fetch(href, {
      method,
      redirect: 'follow',
      headers: { 'user-agent': userAgent },
      signal: AbortSignal.timeout(linkTimeoutMs),
    })

  const results = await Promise.all(
    [...hrefs].map(async (href) => {
      try {
        const head = await request(href, 'HEAD')
        // Some origins refuse HEAD outright; ask again the only other way.
        const response =
          head.status === 405 || head.status === 501
            ? await request(href, 'GET')
            : head

        return { href, status: response.status }
      } catch (error) {
        return { href, status: null, reason: error.message }
      }
    }),
  )

  for (const { href, status, reason } of results) {
    if (status === null) {
      warnings.push(`${href} unreachable (${reason}), not checked`)
    } else if (status >= 400 && status < 600) {
      errors.push(`evidence href ${href} responded ${status}`)
    }
  }
}

async function main() {
  const errors = []
  const warnings = []
  const totals = []

  for (const source of contentSources) {
    const files = getSourceFiles(source, errors)
    totals.push(`${source.name}:${files.length}`)

    for (const relativePath of files) {
      const absolutePath = path.join(frontendRoot, relativePath)
      const rawContent = fs.readFileSync(absolutePath, 'utf-8')
      const parsed = parseFrontmatter(rawContent, relativePath, errors)

      if (!parsed) {
        continue
      }

      validateMetadata(
        parsed.metadata,
        parsed.content,
        source,
        relativePath,
        errors,
      )
    }
  }

  const lists = loadExport(listsFile, 'lists', errors)
  const externalHrefs = new Set()
  const entryCount = validateLists(lists, errors, externalHrefs)
  totals.push(`lists:${lists.length}`, `entries:${entryCount}`)

  const logEvents = loadExport(logFile, 'log', errors)
  validateLog(logEvents, errors, externalHrefs)
  totals.push(`log:${logEvents.length}`)

  const hourTable = loadExport(hoursFile, 'hourTable', errors)
  validateHours(hourTable, errors, lists)
  validateInkAge(errors)

  await checkExternalHrefs(externalHrefs, errors, warnings)

  for (const warning of warnings) {
    console.warn(`! ${warning}`)
  }

  if (errors.length > 0) {
    console.error('Content validation failed:')
    for (const error of errors) {
      console.error(`- ${error}`)
    }
    process.exit(1)
  }

  console.log(`Content validation passed (${totals.join(', ')})`)
}

await main()
