import Link from 'next/link'
import { NotFoundScene } from '@/app/components/NotFoundScene'
import styles from '@/app/components/NotFoundScene.module.css'

export default function NotFound() {
  return (
    <section>
      <h1 className={styles.heading}>404</h1>
      <p className={styles.lead}>
        This page was torn out of the notebook. Whatever was written here went
        with it — page number and all.
      </p>
      <NotFoundScene />
      <p className={`margin-note ${styles.marginNote}`}>
        try /search in the terminal (ctrl+k)
      </p>
      <p className={styles.navRow}>
        <Link href="/">home</Link>
        {' · '}
        <Link href="/projects">projects</Link>
        {' · '}
        <Link href="/writing">writing</Link>
      </p>
    </section>
  )
}
