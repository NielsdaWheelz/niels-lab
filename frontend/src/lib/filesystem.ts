/**
 * Virtual filesystem abstraction for terminal navigation
 * Maps MDX content to a Unix-like directory structure
 */

export type FSNode = {
  type: 'file' | 'directory'
  name: string
  path: string
  content?: string // for files
  children?: FSNode[] // for directories
  metadata?: Record<string, string>
}

export type FileSystem = {
  root: FSNode
  resolve: (path: string, cwd: string) => FSNode | null
  list: (path: string, cwd: string) => FSNode[] | null
  read: (path: string, cwd: string) => string | null
  isDirectory: (path: string, cwd: string) => boolean
  exists: (path: string, cwd: string) => boolean
  completePath: (partial: string, cwd: string) => string[]
}

// normalize path (resolve . and .., handle leading/trailing slashes)
export function normalizePath(path: string, cwd: string = '/'): string {
  // handle relative paths
  const fullPath = path.startsWith('/') ? path : `${cwd}/${path}`

  // split and filter empty segments
  const parts = fullPath.split('/').filter(Boolean)
  const resolved: string[] = []

  for (const part of parts) {
    if (part === '.') continue
    if (part === '..') {
      resolved.pop()
    } else {
      resolved.push(part)
    }
  }

  return '/' + resolved.join('/')
}

// build filesystem from content
export function buildFileSystem(
  blogPosts: Array<{ slug: string; metadata: Record<string, string>; content: string }>,
  projects: Array<{ slug: string; metadata: Record<string, string>; content: string }>,
  braindumps: Array<{ slug: string; metadata: Record<string, string>; content: string }>
): FileSystem {
  const root: FSNode = {
    type: 'directory',
    name: '/',
    path: '/',
    children: [
      {
        type: 'directory',
        name: 'blog',
        path: '/blog',
        children: blogPosts.map(post => ({
          type: 'file' as const,
          name: post.slug,
          path: `/blog/${post.slug}`,
          content: post.content,
          metadata: post.metadata,
        })),
      },
      {
        type: 'directory',
        name: 'projects',
        path: '/projects',
        children: projects.map(project => ({
          type: 'directory' as const,
          name: project.slug,
          path: `/projects/${project.slug}`,
          children: [
            {
              type: 'file' as const,
              name: 'README',
              path: `/projects/${project.slug}/README`,
              content: project.content,
              metadata: project.metadata,
            },
            {
              type: 'file' as const,
              name: 'demo',
              path: `/projects/${project.slug}/demo`,
              content: `# Demo\nRun: play ${project.slug}`,
              metadata: { executable: 'true' },
            },
          ],
        })),
      },
      {
        type: 'directory',
        name: 'braindumps',
        path: '/braindumps',
        children: braindumps.map(dump => ({
          type: 'file' as const,
          name: dump.slug,
          path: `/braindumps/${dump.slug}`,
          content: dump.content,
          metadata: dump.metadata,
        })),
      },
    ],
  }

  function findNode(path: string): FSNode | null {
    const normalized = normalizePath(path, '/')
    if (normalized === '/') return root

    const parts = normalized.split('/').filter(Boolean)
    let current: FSNode = root

    for (const part of parts) {
      if (current.type !== 'directory' || !current.children) return null
      const child = current.children.find(c => c.name === part)
      if (!child) return null
      current = child
    }

    return current
  }

  return {
    root,

    resolve(path: string, cwd: string): FSNode | null {
      const normalized = normalizePath(path, cwd)
      return findNode(normalized)
    },

    list(path: string, cwd: string): FSNode[] | null {
      const node = this.resolve(path, cwd)
      if (!node || node.type !== 'directory') return null
      return node.children || []
    },

    read(path: string, cwd: string): string | null {
      const node = this.resolve(path, cwd)
      if (!node || node.type !== 'file') return null
      return node.content || ''
    },

    isDirectory(path: string, cwd: string): boolean {
      const node = this.resolve(path, cwd)
      return node?.type === 'directory'
    },

    exists(path: string, cwd: string): boolean {
      return this.resolve(path, cwd) !== null
    },

    completePath(partial: string, cwd: string): string[] {
      const normalized = normalizePath(partial, cwd)
      const parts = normalized.split('/')
      const lastPart = parts.pop() || ''
      const parentPath = parts.join('/') || '/'

      const parent = findNode(parentPath)
      if (!parent || parent.type !== 'directory' || !parent.children) return []

      return parent.children
        .filter(c => c.name.startsWith(lastPart))
        .map(c => c.name + (c.type === 'directory' ? '/' : ''))
    },
  }
}

