import './globals.css'
import type { Metadata } from 'next'
import { Newsreader } from 'next/font/google'
import { quattro } from '@/fonts'
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

// opsz axis kept so `font-optical-sizing: auto` has a real axis to drive.
const newsreader = Newsreader({
  subsets: ['latin'],
  style: ['normal', 'italic'],
  axes: ['opsz'],
  variable: '--font-serif',
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

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html
      lang="en"
      className={`${newsreader.variable} ${quattro.variable}`}
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
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  )
}
