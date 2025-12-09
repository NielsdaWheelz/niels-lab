import './globals.css'
import type { Metadata } from 'next'
import { JetBrains_Mono } from 'next/font/google'
import { Navbar } from '@/app/components/nav'
import Footer from '@/app/components/footer'
import { Analytics } from '@vercel/analytics/react'
import { SpeedInsights } from '@vercel/speed-insights/next'
import { baseUrl } from '@/app/sitemap'
import { SketchCanvas } from '@/app/components/SketchCanvas'
import { TerminalWrapper, ContentData } from '@/app/components/Terminal/TerminalWrapper'
import { getBlogPosts } from '@/app/blog/utils'
import { getProjects } from '@/app/projects/utils'
import { getBraindumps } from '@/app/braindumps/utils'
import { getCVContent } from '@/app/cv/utils'

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
})

export const metadata: Metadata = {
  metadataBase: new URL(baseUrl),
  title: {
    default: 'niels',
    template: '%s - niels',
  },
  description: 'welcome to my stuff.',
  openGraph: {
    title: 'niels',
    description: 'welcome to my stuff.',
    url: baseUrl,
    siteName: 'niels',
    locale: 'en_US',
    type: 'website',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
}

function getTerminalData(): ContentData {
  const blogPosts = getBlogPosts().map(p => ({
    slug: p.slug,
    metadata: p.metadata as Record<string, string>,
    content: p.content,
  }))
  
  const projects = getProjects().map(p => ({
    slug: p.slug,
    metadata: p.metadata as Record<string, string>,
    content: p.content,
  }))
  
  const braindumps = getBraindumps().map(p => ({
    slug: p.slug,
    metadata: p.metadata as Record<string, string>,
    content: p.content,
  }))
  
  const cvContent = getCVContent()
  const cv = {
    metadata: cvContent.metadata as Record<string, string>,
    content: cvContent.content,
  }
  
  return { blogPosts, projects, braindumps, cv }
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const terminalData = getTerminalData()
  
  return (
    <html lang="en" className={jetbrainsMono.variable}>
      <body>
        <SketchCanvas />
        <Navbar />
        <main>
          {children}
        </main>
        <Footer />
        <TerminalWrapper data={terminalData} />
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  )
}
