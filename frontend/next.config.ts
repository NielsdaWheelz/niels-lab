import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // the /og route reads these fonts at runtime; without this they are not
  // traced into the serverless function bundle and the route 500s on Vercel
  outputFileTracingIncludes: {
    '/og': ['./src/fonts/Newsreader-*.ttf'],
  },
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'X-Frame-Options', value: 'DENY' },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
        ],
      },
    ]
  },
}

export default nextConfig
