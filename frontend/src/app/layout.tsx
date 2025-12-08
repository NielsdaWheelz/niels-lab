import './globals.css'
import type { Metadata } from 'next'
import { JetBrains_Mono } from 'next/font/google'
import { Navbar } from '@/app/components/nav'
import Footer from '@/app/components/footer'
import { Analytics } from '@vercel/analytics/react'
import { SpeedInsights } from '@vercel/speed-insights/next'
import { baseUrl } from '@/app/sitemap'
import { TerminalWrapper } from '@/app/components/Terminal/TerminalWrapper'
import { SketchCanvas } from '@/app/components/SketchCanvas'
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

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // Build terminal data (server-side)
  const cvContent = getCVContent()
  const terminalData = {
    blogPosts: getBlogPosts(),
    projects: getProjects(),
    braindumps: getBraindumps(),
    cv: cvContent,
  }

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
