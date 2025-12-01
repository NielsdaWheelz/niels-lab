# Terminal Feature Specification

> A terminal emulator for power-user navigation on a brutalist portfolio site.

## Overview

The terminal provides an alternative navigation interface at the bottom of the page. It reads from a virtual filesystem mapped to site content and drives Next.js routing. The terminal is a progressive enhancement—all content remains accessible via normal navigation.

**Core principle**: URL is the source of truth. Terminal cwd is derived from the current pathname, not stored independently.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      layout.tsx (server)                │
│                                                         │
│  - Builds filesystem from getBlogPosts/getProjects/etc  │
│  - Passes filesystem to Terminal as prop                │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │              <main>{children}</main>              │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │            <Terminal filesystem={fs} />           │  │
│  │                                                   │  │
│  │  - Reads cwd from usePathname()                   │  │
│  │  - Writes navigation via router.push()            │  │
│  │  - Manages input, output, history locally         │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

| Flow | Description |
|------|-------------|
| Terminal → Router | `cd` command triggers `router.push()` |
| Router → Terminal | URL changes update terminal cwd via `usePathname()` |
| Filesystem → Terminal | `ls`, `cat` read from static content |
| User → Terminal | Keystrokes, history navigation, tab completion |

### Two-Way Sync

The terminal and browser URL stay in sync automatically:

| User Action | Result |
|-------------|--------|
| Types `cd /projects` | URL and terminal both show `/projects` |
| Clicks nav link to `/blog` | URL and terminal both show `/blog` |
| Uses browser back button | URL and terminal both update |
| Refreshes page | Terminal cwd matches URL |

---

## Filesystem Structure

Virtual filesystem mapped from MDX content:

```
/
├── blog/
│   ├── fractal-week-one        # file (post content)
│   ├── fractal-week-two
│   ├── fractal-week-three
│   ├── fractal-week-four
│   ├── my-first-project
│   ├── zero-to-hero-one
│   └── zero-to-hero-two
├── projects/
│   ├── factory-simulator/      # directory
│   │   └── README              # file (project content)
│   ├── fractal-chat/
│   │   └── README
│   ├── fractal-go/
│   │   └── README
│   ├── makemore/
│   │   └── README
│   └── nexus/
│       └── README
└── braindumps/
    └── 2025-12-01              # file (dump content)
```

**Conventions:**
- Blog posts: files directly under `/blog/`
- Projects: directories containing `README` file
- Braindumps: files directly under `/braindumps/`
- Max depth: 2 levels (e.g., `/projects/makemore/README`)

---

## Commands

### `pwd`
Print working directory.

```
~/projects $ pwd
/projects
```

### `ls [path]`
List directory contents. Directories suffixed with `/`.

```
~ $ ls
blog/  projects/  braindumps/

~/projects $ ls
factory-simulator/  fractal-chat/  fractal-go/  makemore/  nexus/

~ $ ls /blog
fractal-week-one  fractal-week-two  fractal-week-three  ...
```

**Errors:**
- Path not found: `ls: cannot access 'foo': No such file or directory`

**Edge cases:**
- `ls` on a file: shows the filename (like real ls)
- `ls` on empty directory: empty output

### `cd <path>`
Change directory. Triggers page navigation.

```
~ $ cd /projects/makemore
~/projects/makemore $ 
```

Supports:
- Absolute paths: `cd /blog`
- Relative paths: `cd makemore`
- Parent: `cd ..`
- Current: `cd .`
- Home: `cd` or `cd ~` or `cd /`

**Errors:**
- Path not found: `cd: no such file or directory: foo`

**Edge cases:**
- `cd ..` at root: stays at root, no error
- `cd` with no args: navigates to `/`
- `cd` to a file: navigates to that file's page

### `cat <path>`
Display file contents. Shows MDX content body (frontmatter stripped).

```
~/blog $ cat fractal-week-one
* It's been been good so far.
* A few difficult moments, but overall not too hard at all.
...
```

**Errors:**
- Path not found: `cat: foo: No such file or directory`
- Path is directory: `cat: projects: Is a directory`

### `clear`
Clear terminal output.

### `help`
Display command reference.

```
~ $ help
commands:
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
  escape        blur terminal
```

---

## Tab Completion

### Behavior
- **Single Tab**: completes if exactly one match
- **Double Tab**: shows all matches if multiple

### Completable Items
- Command names: `l<TAB>` → `ls`
- Paths: `/pro<TAB>` → `/projects/`

### Examples

```
# Unambiguous - completes immediately
$ l<TAB>
$ ls

# Ambiguous - shows options on double tab
$ c<TAB>
$ c<TAB><TAB>
cat  cd  clear

# Path completion
$ cd /pro<TAB>
$ cd /projects/

$ cd /projects/m<TAB>
$ cd /projects/makemore/
```

---

## Keyboard Shortcuts

| Key | Action | Context |
|-----|--------|---------|
| `Ctrl+K` | Focus terminal | Global |
| `Escape` | Blur terminal | Terminal focused |
| `Enter` | Execute command | Terminal focused |
| `↑` | Previous command in history | Terminal focused |
| `↓` | Next command in history | Terminal focused |
| `Tab` | Autocomplete | Terminal focused |

---

## History

- Navigate with `↑` and `↓` arrow keys
- Stored in `localStorage` under key `terminal-history`
- Maximum 100 entries
- Consecutive duplicates not stored
- Persists across sessions

---

## Visual Design

### Layout

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│                    Page Content                         │
│                                                         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ ~/projects $ ls                                         │
│ factory-simulator/  fractal-go/  makemore/  nexus/     │
│ ~/projects $ _                                          │
└─────────────────────────────────────────────────────────┘
```

### Dimensions
- Position: fixed to viewport bottom
- Width: 100%
- Height: 140px
- z-index: 50

### Colors
| Element | Color |
|---------|-------|
| Background | `#0a0a0a` |
| Text | `#e0e0e0` |
| Prompt path | `#666` |
| Prompt `$` | `#e0e0e0` |
| Directories | `#67e8f9` (cyan) |
| Errors | `#f87171` (red) |
| Border top | `#222` |

### Prompt Format
```
~/path/here $ 
```
- `~` represents root `/`
- Current directory path shown
- `$` separator
- Space after `$`

### Focus Indicator
- Blinking cursor in input when focused
- Border top color: `#222` → `#444` when focused

### Mobile
- Hidden on viewport width < 768px
- Use CSS: `@media (max-width: 767px) { display: none; }`

---

## State Management

### Derived State (not stored)
```typescript
cwd = usePathname()  // always matches URL
```

### Persisted State
```typescript
// localStorage key: 'terminal-history'
history: string[] = []  // max 100 entries
```

### Component State (ephemeral)
```typescript
input: string = ""
output: OutputLine[] = []  // max 100 lines
historyIndex: number = -1  // -1 = not navigating history
focused: boolean = false
```

### OutputLine Type
```typescript
type OutputLine = {
  type: 'command' | 'output' | 'error'
  content: string
}
```

---

## Error Handling

| Scenario | Message |
|----------|---------|
| Unknown command | `command not found: foo` |
| Path not found (ls) | `ls: cannot access 'foo': No such file or directory` |
| Path not found (cd) | `cd: no such file or directory: foo` |
| Path not found (cat) | `cat: foo: No such file or directory` |
| cat on directory | `cat: projects: Is a directory` |
| Empty input | No-op, show new prompt |
| Whitespace-only input | No-op, show new prompt |

---

## File Structure

```
src/
├── app/
│   ├── layout.tsx              # Renders <Terminal /> with filesystem prop
│   └── components/
│       └── Terminal/
│           ├── index.tsx       # Main terminal component
│           ├── useTerminal.ts  # State management hook
│           └── commands.ts     # Command implementations
└── lib/
    └── filesystem.ts           # Virtual filesystem (already exists)
```

---

## Integration

### layout.tsx Changes

```tsx
// Server component - can call getBlogPosts etc.
import { Terminal } from '@/app/components/Terminal'
import { buildFileSystem } from '@/lib/filesystem'
import { getBlogPosts } from '@/app/blog/utils'
import { getProjects } from '@/app/projects/utils'
import { getBraindumps } from '@/app/braindumps/utils'

export default function RootLayout({ children }) {
  const filesystem = buildFileSystem(
    getBlogPosts(),
    getProjects(),
    getBraindumps()
  )

  return (
    <html>
      <body>
        <Navbar />
        <main>{children}</main>
        <Footer />
        <Terminal filesystem={filesystem} />
      </body>
    </html>
  )
}
```

---

## Acceptance Criteria

- [ ] Terminal visible at bottom of viewport (fixed position)
- [ ] Hidden on mobile (< 768px)
- [ ] `Ctrl+K` focuses terminal from anywhere on page
- [ ] `Escape` blurs terminal
- [ ] `pwd` prints current directory
- [ ] `ls` lists current directory contents
- [ ] `ls <path>` lists specified directory
- [ ] `cd <path>` navigates page and updates URL
- [ ] `cat <file>` displays file content
- [ ] `clear` clears output
- [ ] `help` shows command reference
- [ ] Tab completion works for commands
- [ ] Tab completion works for paths
- [ ] Single tab completes unambiguous matches
- [ ] Double tab shows multiple matches
- [ ] ↑/↓ navigates command history
- [ ] History persists across page refreshes
- [ ] cwd stays in sync with URL on all navigation types
- [ ] Error messages display in red
- [ ] Directories display in cyan
- [ ] No console errors

---

## Out of Scope

Explicitly NOT building:
- `play` command (deferred until demos integrated)
- `grep`, `find`, `head`, `tail`
- Pipes and redirects (`|`, `>`, `>>`)
- Background processes
- Multiple terminals
- Terminal resizing
- Mobile support
- Vim/nano/any editor
- File creation/deletion
- Easter eggs (future enhancement)

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Tab completion edge cases | Medium | Low | Start simple, iterate |
| Large filesystem slows page | Low | Medium | Filesystem is small (~50 nodes) |
| History sync issues | Low | Low | localStorage is reliable |
| Focus conflicts with page | Medium | Medium | Only capture when explicitly focused |

---

## Future Enhancements (Not This Phase)

- `play <project>` - Launch embedded demo
- Easter eggs (`rm -rf /`, `sudo`, `exit`)
- `grep` for searching content
- Command aliases
- Themes
- Export history

