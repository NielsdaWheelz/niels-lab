import './globals.css'
import type { Metadata } from 'next'
import { GeistMono } from 'geist/font/mono'
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
import {
  TerminalWrapper,
  ContentData,
} from '@/app/components/Terminal/TerminalWrapper'
import { getProjects } from '@/app/projects/utils'
import { getWritingPosts } from '@/app/writing/utils'
import { getCVContent } from '@/app/cv/utils'

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

  const cvContent = getCVContent()
  const cv = {
    metadata: cvContent.metadata as Record<string, string>,
    content: cvContent.content,
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
    <html lang="en" className={GeistMono.variable}>
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
