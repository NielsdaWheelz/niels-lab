import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { ImageResponse } from 'next/og'
import { getSiteUrl, siteDescription, siteName } from '@/app/site'

export const runtime = 'nodejs'

// satori has no CSS custom properties: these mirror the dark-theme tokens in
// globals.css and must be changed with them. The card is always night writing.
const ink = '#16161d'
const murasaki = '#a87bc4'
const celadon = '#7fa99b'
const faded = '#8f8a99'
const rule = '#2a2833'

// read at module scope so the bytes are cached between invocations; the files
// reach the serverless bundle via outputFileTracingIncludes in next.config.ts.
const newsreaderRegular = readFile(
  path.join(process.cwd(), 'src', 'fonts', 'Newsreader-Regular.ttf'),
)
const newsreaderItalic = readFile(
  path.join(process.cwd(), 'src', 'fonts', 'Newsreader-Italic.ttf'),
)

function titleFontSize(title: string) {
  const len = title.length
  if (len <= 18) return 92
  if (len <= 32) return 72
  if (len <= 50) return 58
  if (len <= 70) return 46
  return 38
}

export async function GET(request: Request) {
  const url = new URL(request.url)
  const title = url.searchParams.get('title') || siteName
  const description =
    url.searchParams.get('description')?.slice(0, 220) || siteDescription
  const siteHost = new URL(getSiteUrl()).host
  const [regularData, italicData] = await Promise.all([
    newsreaderRegular,
    newsreaderItalic,
  ])

  return new ImageResponse(
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        width: '100%',
        height: '100%',
        backgroundColor: ink,
        fontFamily: 'Newsreader',
        padding: '72px 88px',
      }}
    >
      <div style={{ display: 'flex', fontSize: 26, color: celadon }}>
        {siteHost}
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 28,
          borderTop: `1px solid ${rule}`,
          paddingTop: 44,
        }}
      >
        <div
          style={{
            display: 'flex',
            fontSize: titleFontSize(title),
            fontStyle: 'italic',
            lineHeight: 1.15,
            color: murasaki,
            maxWidth: 1000,
          }}
        >
          {title}
        </div>
        <div
          style={{
            display: 'flex',
            fontSize: 26,
            lineHeight: 1.45,
            color: faded,
            maxWidth: 880,
          }}
        >
          {description}
        </div>
      </div>
    </div>,
    {
      width: 1200,
      height: 630,
      fonts: [
        {
          name: 'Newsreader',
          data: regularData,
          weight: 400,
          style: 'normal',
        },
        {
          name: 'Newsreader',
          data: italicData,
          weight: 400,
          style: 'italic',
        },
      ],
    },
  )
}
