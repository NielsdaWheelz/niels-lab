import { FileSystem, normalizePath } from '@/lib/filesystem'

export type CommandResult = {
  output: string
  isError: boolean
  navigate?: string // path to navigate to
}

export type CommandContext = {
  fs: FileSystem
  cwd: string
}

const HELP_TEXT = `commands:
  pwd           print working directory
  ls [path]     list directory contents
  cd <path>     change directory (navigates page)
  cat <file>    display file contents
  clear         clear terminal output
  help          show this message

keyboard:
  ctrl+k        focus terminal
  tab           autocomplete
  ↑/↓           command history
  escape        blur terminal`

const COMMANDS = ['pwd', 'ls', 'cd', 'cat', 'clear', 'help']

export function executeCommand(
  input: string,
  ctx: CommandContext
): CommandResult {
  const trimmed = input.trim()
  if (!trimmed) {
    return { output: '', isError: false }
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

