import './globals.css'
import type { Metadata } from 'next'
import { GeistMono } from 'geist/font/mono'
import { GeistSans } from 'geist/font/sans'
import { Newsreader, Caveat } from 'next/font/google'
import { themeInitScript } from '@/lib/theme'
import { Navbar } from '@/app/components/nav'
import Footer from '@/app/components/footer'
import { Analytics } from '@vercel/analytics/react'
import { SpeedInsights } from '@vercel/speed-insights/next'
import {
  createPageMetadata,
  getSiteUrl,
  shouldIndexSite,
  siteDescription,
  siteName,
} from '@/app/site'
import { StructuredData } from '@/app/components/StructuredData'
import {
  TerminalWrapper,
  ContentData,
} from '@/app/components/Terminal/TerminalWrapper'
import { getProjects } from '@/app/projects/utils'
import { getWritingPosts } from '@/app/writing/utils'

const newsreader = Newsreader({
  subsets: ['latin'],
  style: ['normal', 'italic'],
  variable: '--font-serif',
  display: 'swap',
})

const caveat = Caveat({
  subsets: ['latin'],
  variable: '--font-hand',
  display: 'swap',
})

const siteUrl = getSiteUrl()
const shouldIndex = shouldIndexSite()
const rootMetadata = createPageMetadata({
  title: `${siteName} — AI Systems Engineer`,
  description: siteDescription,
  path: '/',
})

export const metadata: Metadata = {
  ...rootMetadata,
  metadataBase: new URL(siteUrl),
  title: {
    default: `${siteName} — AI Systems Engineer`,
    template: `%s — ${siteName}`,
  },
  robots: {
    index: shouldIndex,
    follow: shouldIndex,
    googleBot: {
      index: shouldIndex,
      follow: shouldIndex,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
}

function getTerminalData(): ContentData {
  const writingPosts = getWritingPosts().map((p) => ({
    slug: p.slug,
    metadata: p.metadata as Record<string, string>,
    content: `${p.metadata.summary}\n\nRead the full note at /writing/${p.slug}`,
  }))

  const projects = getProjects().map((p) => ({
    slug: p.slug,
    metadata: p.metadata as Record<string, string>,
    content: `${p.metadata.summary}\n\nInspect the full case study at /projects/${p.slug}`,
  }))

  const cv = {
    metadata: {
      title: 'CV',
      publishedAt: '2025-12-01',
      summary: 'Professional experience and background',
    },
    content:
      'Software engineer working across AI systems, deterministic backends, and product interfaces.\n\nRead the full background at /cv',
  }

  return { writingPosts, projects, cv }
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const terminalData = getTerminalData()

  return (
    <html
      lang="en"
      className={`${GeistSans.variable} ${GeistMono.variable} ${newsreader.variable} ${caveat.variable}`}
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
        <StructuredData />
      </head>
      <body>
        <a href="#main-content" className="skip-link">
          Skip to content
        </a>
        <Navbar />
        <main id="main-content">{children}</main>
        <Footer />
        <TerminalWrapper data={terminalData} />
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  )
}
