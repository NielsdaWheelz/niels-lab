import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { ImageResponse } from 'next/og'
import { getSiteUrl, siteDescription, siteName } from '@/app/site'

export const runtime = 'nodejs'

const geistMonoFont = readFile(
  path.join(
    process.cwd(),
    'node_modules',
    'geist',
    'dist',
    'fonts',
    'geist-mono',
    'GeistMono-Bold.ttf',
  ),
)

export async function GET(request: Request) {
  const url = new URL(request.url)
  const title = url.searchParams.get('title') || siteName
  const siteHost = new URL(getSiteUrl()).host
  const fontData = await geistMonoFont

  return new ImageResponse(
    <div
      style={{
        display: 'flex',
        width: '100%',
        height: '100%',
        padding: '72px',
        background: '#fdf6e3',
        color: '#3d3d3d',
        flexDirection: 'column',
        justifyContent: 'space-between',
        fontFamily: 'Geist Mono',
      }}
    >
      <div style={{ fontSize: 28, color: '#1a7080' }}>{siteHost}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <div style={{ fontSize: 60, fontWeight: 700, lineHeight: 1.1 }}>
          {title}
        </div>
        <div style={{ fontSize: 28, color: '#6b6b6b' }}>{siteDescription}</div>
      </div>
    </div>,
    {
      width: 1200,
      height: 630,
      fonts: [
        {
          name: 'Geist Mono',
          data: fontData,
          weight: 700,
          style: 'normal',
        },
      ],
    },
  )
}
