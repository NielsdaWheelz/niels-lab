import './globals.css'
import type { Metadata } from 'next'
import { GeistMono } from 'geist/font/mono'
import { Newsreader, Caveat } from 'next/font/google'
import { themeInitScript } from '@/lib/theme'
import { Navbar } from '@/app/components/nav'
import Footer from '@/app/components/footer'
import { Analytics } from '@vercel/analytics/react'
import { SpeedInsights } from '@vercel/speed-insights/next'
import {
  getOgImageUrl,
  getSiteUrl,
  shouldIndexSite,
  siteDescription,
  siteName,
} from '@/app/site'
import { SketchCanvas } from '@/app/components/SketchCanvas'
import { StructuredData } from '@/app/components/StructuredData'
import {
  TerminalWrapper,
  ContentData,
} from '@/app/components/Terminal/TerminalWrapper'
import { getProjects } from '@/app/projects/utils'
import { getWritingPosts } from '@/app/writing/utils'
import { entries as cvEntries, skills as cvSkills } from '@/app/cv/data'

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
const ogImageUrl = getOgImageUrl(siteName)

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: siteName,
    template: `%s | ${siteName}`,
  },
  description: siteDescription,
  alternates: {
    canonical: '/',
  },
  openGraph: {
    title: siteName,
    description: siteDescription,
    url: '/',
    siteName,
    locale: 'en_US',
    type: 'website',
    images: [
      {
        url: ogImageUrl,
        width: 1200,
        height: 630,
        alt: siteName,
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: siteName,
    description: siteDescription,
    images: [ogImageUrl],
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
    content: p.content,
  }))

  const projects = getProjects().map((p) => ({
    slug: p.slug,
    metadata: p.metadata as Record<string, string>,
    content: p.content,
  }))

  const cv = {
    metadata: {
      title: 'CV',
      publishedAt: '2025-12-01',
      summary: 'Professional experience and background',
    },
    content:
      cvEntries
        .map(
          (e) =>
            `${e.title}${'subtitle' in e && e.subtitle ? ' – ' + e.subtitle : ''}\n${e.date}${'bullets' in e && e.bullets ? '\n' + e.bullets.map((b) => '- ' + b).join('\n') : ''}`,
        )
        .join('\n\n') +
      '\n\nSKILLS\n' +
      Object.entries(cvSkills)
        .map(([k, v]) => `${k}: ${v.join(', ')}`)
        .join('\n'),
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
      className={`${GeistMono.variable} ${newsreader.variable} ${caveat.variable}`}
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
        <StructuredData />
      </head>
      <body>
        <SketchCanvas />
        <Navbar />
        <main>{children}</main>
        <Footer />
        <TerminalWrapper data={terminalData} />
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  )
}
