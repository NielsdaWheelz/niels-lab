import fs from 'fs'
import path from 'path'

type Metadata = {
  title: string
  publishedAt: string
  summary: string
}

function parseFrontmatter(fileContent: string) {
  const frontmatterRegex = /---\s*([\s\S]*?)\s*---/
  const match = frontmatterRegex.exec(fileContent)

  if (!match) {
    throw new Error('Invalid MDX file: missing frontmatter delimiters (---)')
  }

  const frontMatterBlock = match[1]
  const content = fileContent.replace(frontmatterRegex, '').trim()
  const frontMatterLines = frontMatterBlock.trim().split('\n')
  const metadata: Partial<Metadata> = {}

  frontMatterLines.forEach((line) => {
    const [key, ...valueArr] = line.split(': ')
    let value = valueArr.join(': ').trim()
    value = value.replace(/^['"](.*)['"]$/, '$1')
    metadata[key.trim() as keyof Metadata] = value
  })

  return { metadata: metadata as Metadata, content }
}

export function getCVContent() {
  const cvPath = path.join(process.cwd(), 'src', 'app', 'cv', 'cv.mdx')
  const rawContent = fs.readFileSync(cvPath, 'utf-8')
  return parseFrontmatter(rawContent)
}
