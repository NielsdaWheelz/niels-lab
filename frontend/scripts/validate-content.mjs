#!/usr/bin/env node

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const frontendRoot = path.resolve(__dirname, '..')
const frontmatterRegex = /---\s*([\s\S]*?)\s*---/
const publishedAtRegex = /^\d{4}-\d{2}-\d{2}$/

const contentSources = [
  {
    name: 'projects',
    directory: 'src/app/projects/posts',
    requiredKeys: ['title', 'publishedAt', 'summary', 'image'],
  },
  {
    name: 'writing',
    directory: 'src/app/writing/posts',
    requiredKeys: ['title', 'publishedAt', 'summary'],
  },
  {
    name: 'cv',
    file: 'src/app/cv/cv.mdx',
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

  if (!content) {
    errors.push(`${relativePath}: content body is empty`)
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

function validateImagePath(value, relativePath, errors) {
  if (!value.trim()) {
    errors.push(`${relativePath}: image must be non-empty`)
    return
  }

  if (!value.startsWith('/')) {
    errors.push(`${relativePath}: image must be a root-relative public path`)
    return
  }

  const imagePath = path.join(frontendRoot, 'public', value.slice(1))
  if (!fs.existsSync(imagePath)) {
    errors.push(`${relativePath}: image file is missing at public${value}`)
  }
}

function validateMetadata(metadata, source, relativePath, errors) {
  for (const key of source.requiredKeys) {
    if (!Object.prototype.hasOwnProperty.call(metadata, key)) {
      errors.push(`${relativePath}: missing required frontmatter key "${key}"`)
    }
  }

  if (typeof metadata.title !== 'string' || metadata.title.trim() === '') {
    errors.push(`${relativePath}: title must be present and non-empty`)
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

  if (Object.prototype.hasOwnProperty.call(metadata, 'image')) {
    validateImagePath(metadata.image, relativePath, errors)
  }

  for (const fieldName of ['repoUrl', 'liveUrl']) {
    if (
      typeof metadata[fieldName] === 'string' &&
      metadata[fieldName].trim() !== ''
    ) {
      validateUrl(metadata[fieldName], fieldName, relativePath, errors)
    }
  }
}

function getSourceFiles(source, errors) {
  if (source.file) {
    const absoluteFile = path.join(frontendRoot, source.file)
    if (!fs.existsSync(absoluteFile)) {
      errors.push(`${source.file}: content file is missing`)
      return []
    }

    return [source.file]
  }

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

function main() {
  const errors = []
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

      validateMetadata(parsed.metadata, source, relativePath, errors)
    }
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

main()
