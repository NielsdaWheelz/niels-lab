import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { ImageResponse } from 'next/og'
import { getSiteUrl, siteDescription, siteName } from '@/app/site'

export const runtime = 'nodejs'

const geistMonoBold = readFile(
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

const geistMonoRegular = readFile(
  path.join(
    process.cwd(),
    'node_modules',
    'geist',
    'dist',
    'fonts',
    'geist-mono',
    'GeistMono-Regular.ttf',
  ),
)

const cream = '#fbf4e2'
const ink = '#2e2a23'
const mutedInk = '#6f675a'
const terracotta = '#d4552a'
const teal = '#17697a'
const ochre = '#c0932f'
const borderColor = '#e4d9bf'

const gridBackground = [
  'repeating-linear-gradient(to right, rgba(23,105,122,0.055) 0px, rgba(23,105,122,0.055) 1px, transparent 1px, transparent 32px)',
  'repeating-linear-gradient(to bottom, rgba(23,105,122,0.055) 0px, rgba(23,105,122,0.055) 1px, transparent 1px, transparent 32px)',
].join(', ')

function titleFontSize(title: string) {
  const len = title.length
  if (len <= 18) return 88
  if (len <= 32) return 68
  if (len <= 50) return 54
  if (len <= 70) return 44
  return 36
}

function underlineWidth(title: string, fontSize: number) {
  const firstWord = title.split(' ')[0] ?? title
  const estimate = firstWord.length * fontSize * 0.6
  return Math.max(72, Math.min(260, estimate))
}

export async function GET(request: Request) {
  const url = new URL(request.url)
  const title = url.searchParams.get('title') || siteName
  const description =
    url.searchParams.get('description')?.slice(0, 220) || siteDescription
  const siteHost = new URL(getSiteUrl()).host
  const [boldData, regularData] = await Promise.all([
    geistMonoBold,
    geistMonoRegular,
  ])
  const fontSize = titleFontSize(title)

  return new ImageResponse(
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        width: '100%',
        height: '100%',
        position: 'relative',
        backgroundColor: cream,
        backgroundImage: gridBackground,
        color: ink,
        fontFamily: 'Geist Mono',
        padding: '56px 64px',
      }}
    >
      {/* drafting-sheet frame */}
      <div
        style={{
          position: 'absolute',
          top: 24,
          left: 24,
          right: 24,
          bottom: 24,
          border: `1px solid ${borderColor}`,
        }}
      />
      {/* corner marks */}
      <div
        style={{
          position: 'absolute',
          top: 24,
          left: 24,
          width: 28,
          height: 28,
          borderTop: `2px solid ${teal}`,
          borderLeft: `2px solid ${teal}`,
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: 24,
          right: 24,
          width: 28,
          height: 28,
          borderTop: `2px solid ${teal}`,
          borderRight: `2px solid ${teal}`,
        }}
      />
      <div
        style={{
          position: 'absolute',
          bottom: 24,
          left: 24,
          width: 28,
          height: 28,
          borderBottom: `2px solid ${teal}`,
          borderLeft: `2px solid ${teal}`,
        }}
      />
      <div
        style={{
          position: 'absolute',
          bottom: 24,
          right: 24,
          width: 28,
          height: 28,
          borderBottom: `2px solid ${teal}`,
          borderRight: `2px solid ${teal}`,
        }}
      />

      {/* top row: terminal path + stamp badge */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
        }}
      >
        <div
          style={{
            display: 'flex',
            fontSize: 24,
            fontWeight: 700,
            color: teal,
            letterSpacing: '0.5px',
          }}
        >
          {siteHost}
          {' ~ $ ▌'}
        </div>
        <div
          style={{
            display: 'flex',
            border: `2px solid ${terracotta}`,
            padding: '10px 18px',
            transform: 'rotate(-4deg)',
            fontSize: 15,
            fontWeight: 700,
            color: terracotta,
            letterSpacing: '2px',
          }}
        >
          {"ENGINEER'S NOTEBOOK"}
        </div>
      </div>

      {/* middle: title + underline stroke + description */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '20px',
          maxWidth: '1000px',
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '14px',
          }}
        >
          <div
            style={{
              display: 'flex',
              fontSize,
              fontWeight: 700,
              lineHeight: 1.15,
              letterSpacing: '-1px',
              color: ink,
            }}
          >
            {title}
          </div>
          <div
            style={{
              display: 'flex',
              width: underlineWidth(title, fontSize),
              height: 10,
              backgroundColor: terracotta,
              transform: 'rotate(-1deg)',
            }}
          />
        </div>
        <div
          style={{
            display: 'flex',
            fontSize: 22,
            fontWeight: 400,
            color: mutedInk,
            maxWidth: '760px',
          }}
        >
          {description}
        </div>
      </div>

      {/* bottom row: description tag + page stamp */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'flex-end',
          justifyContent: 'space-between',
        }}
      >
        <div
          style={{
            display: 'flex',
            fontSize: 15,
            fontWeight: 400,
            color: mutedInk,
            letterSpacing: '1px',
          }}
        >
          {'// notes on building things'}
        </div>
        <div
          style={{
            display: 'flex',
            fontSize: 14,
            fontWeight: 700,
            color: ochre,
            letterSpacing: '3px',
          }}
        >
          {'SHEET 01 · REV A'}
        </div>
      </div>
    </div>,
    {
      width: 1200,
      height: 630,
      fonts: [
        {
          name: 'Geist Mono',
          data: boldData,
          weight: 700,
          style: 'normal',
        },
        {
          name: 'Geist Mono',
          data: regularData,
          weight: 400,
          style: 'normal',
        },
      ],
    },
  )
}
