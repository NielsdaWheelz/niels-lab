import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // the /og route reads these fonts at runtime; without this they are not
  // traced into the serverless function bundle and the route 500s on Vercel
  outputFileTracingIncludes: {
    '/og': ['./node_modules/geist/dist/fonts/geist-mono/*.ttf'],
  },
}

export default nextConfig
