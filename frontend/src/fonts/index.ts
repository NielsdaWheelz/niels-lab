import localFont from 'next/font/local'

// iA Writer Quattro (OFL, vendored from iaolo/iA-Fonts): dates, counts, evidence
// chrome, ledger rows. Not a reading face — see docs/pillow-book-spec.md §6.
export const quattro = localFont({
  src: [
    {
      path: './iAWriterQuattroS-Regular.woff2',
      weight: '400',
      style: 'normal',
    },
    {
      path: './iAWriterQuattroS-Italic.woff2',
      weight: '400',
      style: 'italic',
    },
    {
      path: './iAWriterQuattroS-Bold.woff2',
      weight: '700',
      style: 'normal',
    },
  ],
  variable: '--font-quattro',
  display: 'swap',
})
