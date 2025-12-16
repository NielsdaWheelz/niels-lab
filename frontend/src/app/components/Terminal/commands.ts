import { FileSystem, normalizePath } from '@/lib/filesystem'

export type CommandResult = {
  output: string | React.ReactNode
  isError: boolean
  navigate?: string // path to navigate to
  status?: string // optional status line message
}

export type CommandContext = {
  fs: FileSystem
  cwd: string
}

const HELP_TEXT = `commands:
  pwd             print working directory
  ls [path]       list directory contents
  cd <path>       change directory (navigates page)
  cat <file>      display file contents
  search <query>  search content (alias: grep)
  whois           display persona card
  visualize       generate decorative scribble
  clear           clear terminal output
  help            show this message

keyboard:
  ctrl+k          focus terminal
  tab             autocomplete
  ↑/↓             command history
  escape          blur terminal`

const COMMANDS = ['pwd', 'ls', 'cd', 'cat', 'clear', 'help', 'search', 'grep', 'whois', 'visualize', 'sudo']

export function executeCommand(
  input: string,
  ctx: CommandContext
): CommandResult {
  const trimmed = input.trim()
  if (!trimmed) {
    return { output: '', isError: false }
  }

  // handle sudo first
  if (trimmed.startsWith('sudo ')) {
    return cmdSudo(trimmed)
  }

  const parts = trimmed.split(/\s+/)
  const cmd = parts[0]
  const args = parts.slice(1)

  switch (cmd) {
    case 'pwd':
      return cmdPwd(ctx)
    case 'ls':
      return cmdLs(args[0], ctx)
    case 'cd':
      return cmdCd(args[0], ctx)
    case 'cat':
      return cmdCat(args[0], ctx)
    case 'clear':
      return { output: '__CLEAR__', isError: false }
    case 'help':
      return { output: HELP_TEXT, isError: false }
    case 'search':
    case 'grep':
      return cmdSearch(args, ctx)
    case 'whois':
      return cmdWhois()
    case 'visualize':
      return cmdVisualize()
    case 'sudo':
      return cmdSudo(trimmed)
    default:
      return { output: `command not found: ${cmd}`, isError: true }
  }
}

function cmdPwd(ctx: CommandContext): CommandResult {
  return { output: ctx.cwd, isError: false }
}

function cmdLs(path: string | undefined, ctx: CommandContext): CommandResult {
  const targetPath = path || ctx.cwd
  const node = ctx.fs.resolve(targetPath, ctx.cwd)

  if (!node) {
    return {
      output: `ls: cannot access '${targetPath}': No such file or directory`,
      isError: true,
    }
  }

  // ls on a file shows the filename
  if (node.type === 'file') {
    return { output: node.name, isError: false }
  }

  // ls on a directory
  const children = node.children || []
  if (children.length === 0) {
    return { output: '', isError: false }
  }

  const items = children.map(c =>
    c.type === 'directory' ? `${c.name}/` : c.name
  )
  return { output: items.join('  '), isError: false }
}

function cmdCd(path: string | undefined, ctx: CommandContext): CommandResult {
  // cd with no args goes to root
  const targetPath = path || '/'

  // handle ~ as root
  const normalizedInput = targetPath === '~' ? '/' : targetPath
  const resolved = normalizePath(normalizedInput, ctx.cwd)
  const node = ctx.fs.resolve(resolved, '/')

  if (!node) {
    return {
      output: `cd: no such file or directory: ${targetPath}`,
      isError: true,
    }
  }

  // navigate to the path (works for both files and directories)
  return { output: '', isError: false, navigate: resolved }
}

function cmdCat(path: string | undefined, ctx: CommandContext): CommandResult {
  if (!path) {
    return { output: 'cat: missing operand', isError: true }
  }

  const node = ctx.fs.resolve(path, ctx.cwd)

  if (!node) {
    return {
      output: `cat: ${path}: No such file or directory`,
      isError: true,
    }
  }

  if (node.type === 'directory') {
    return {
      output: `cat: ${path}: Is a directory`,
      isError: true,
    }
  }

  return { output: node.content || '', isError: false }
}

// Search/grep command - fuzzy search across content
function cmdSearch(args: string[], ctx: CommandContext): CommandResult {
  // Parse flags
  let showLineNumbers = false
  let limit = 5
  const queryParts: string[] = []

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]
    if (arg === '-n') {
      showLineNumbers = true
    } else if (arg === '-l' && args[i + 1]) {
      limit = parseInt(args[i + 1], 10) || 5
      i++ // skip next arg
    } else if (!arg.startsWith('-')) {
      queryParts.push(arg)
    }
  }

  const query = queryParts.join(' ').toLowerCase()

  if (!query) {
    return { output: 'usage: search <query> [-n] [-l <limit>]', isError: true }
  }

  // Search through filesystem content
  const results: Array<{
    title: string
    path: string
    snippet: string
    lineNum?: number
  }> = []

  function searchNode(node: { type: string; name: string; path: string; content?: string; metadata?: Record<string, string>; children?: typeof node[] }) {
    if (node.type === 'file' && node.content) {
      const content = node.content.toLowerCase()
      const title = node.metadata?.title || node.name

      if (content.includes(query) || title.toLowerCase().includes(query)) {
        // Find snippet with match
        const lines = node.content.split('\n')
        let snippetLine = ''
        let lineNum = 0

        for (let i = 0; i < lines.length; i++) {
          if (lines[i].toLowerCase().includes(query)) {
            snippetLine = lines[i].trim()
            lineNum = i + 1
            break
          }
        }

        // Truncate snippet if too long
        if (snippetLine.length > 80) {
          const idx = snippetLine.toLowerCase().indexOf(query)
          const start = Math.max(0, idx - 30)
          const end = Math.min(snippetLine.length, idx + query.length + 30)
          snippetLine = (start > 0 ? '...' : '') + snippetLine.slice(start, end) + (end < snippetLine.length ? '...' : '')
        }

        results.push({
          title,
          path: node.path,
          snippet: snippetLine || title,
          lineNum: showLineNumbers ? lineNum : undefined,
        })
      }
    }

    if (node.children) {
      node.children.forEach(searchNode)
    }
  }

  searchNode(ctx.fs.root)

  if (results.length === 0) {
    return { output: `no matches for '${queryParts.join(' ')}'`, isError: false }
  }

  // Return structured data for React rendering
  // The output will be processed by the terminal to render as React nodes
  const limitedResults = results.slice(0, limit)

  return {
    output: { __type: 'search', query: queryParts.join(' '), results: limitedResults } as unknown as string,
    isError: false,
    status: `found ${results.length} result${results.length === 1 ? '' : 's'}`,
  }
}

// Whois command - persona card
function cmdWhois(): CommandResult {
  return {
    output: { __type: 'whois' } as unknown as string,
    isError: false,
  }
}

// Visualize command - decorative SVG
function cmdVisualize(): CommandResult {
  // Generate a seed from current timestamp for deterministic randomness
  const seed = Date.now()
  return {
    output: { __type: 'visualize', seed } as unknown as string,
    isError: false,
  }
}

// Sudo easter egg
function cmdSudo(input: string): CommandResult {
  const afterSudo = input.slice(5).trim()

  if (afterSudo === 'make me a sandwich') {
    return {
      output: "what? make it yourself.",
      isError: true,
    }
  }

  return {
    output: "user is not in the sudoers file. this incident will be reported.",
    isError: true,
  }
}

export function completeInput(
  input: string,
  ctx: CommandContext
): { completed: string; options: string[] } {
  const trimmed = input.trimStart()
  const parts = trimmed.split(/\s+/)

  // completing command name
  if (parts.length === 1 && !input.endsWith(' ')) {
    const partial = parts[0]
    const matches = COMMANDS.filter(c => c.startsWith(partial))

    if (matches.length === 1) {
      return { completed: matches[0] + ' ', options: [] }
    }
    return { completed: input, options: matches }
  }

  // completing path argument
  if (parts.length >= 1) {
    const cmd = parts[0]
    const pathArg = parts.length > 1 ? parts[parts.length - 1] : ''

    // only complete paths for commands that take paths
    if (!['ls', 'cd', 'cat'].includes(cmd)) {
      return { completed: input, options: [] }
    }

    const options = ctx.fs.completePath(pathArg, ctx.cwd)

    if (options.length === 1) {
      // single match - complete it
      const beforePath = parts.slice(0, -1).join(' ')
      const prefix = beforePath ? beforePath + ' ' : cmd + ' '

      // figure out the base path
      const pathParts = pathArg.split('/')
      pathParts.pop()
      const basePath = pathParts.length > 0 ? pathParts.join('/') + '/' : ''

      return {
        completed: prefix + basePath + options[0],
        options: []
      }
    }

    return { completed: input, options }
  }

  return { completed: input, options: [] }
}
