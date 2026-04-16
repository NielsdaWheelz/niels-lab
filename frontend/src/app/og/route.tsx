import { ImageResponse } from 'next/og'
import { siteDescription, siteName } from '@/app/site'

export function GET(request: Request) {
  const url = new URL(request.url)
  const title = url.searchParams.get('title') || siteName

  return new ImageResponse(
    (
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
          fontFamily: 'JetBrains Mono, monospace',
        }}
      >
        <div style={{ fontSize: 28, color: '#1a7080' }}>niels.dev</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div style={{ fontSize: 60, fontWeight: 700, lineHeight: 1.1 }}>
            {title}
          </div>
          <div style={{ fontSize: 28, color: '#6b6b6b' }}>
            {siteDescription}
          </div>
        </div>
      </div>
    ),
    {
      width: 1200,
      height: 630,
    }
  )
}
