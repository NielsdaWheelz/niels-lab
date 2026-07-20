import Link from 'next/link'
import { createPageMetadata } from '@/app/site'
import styles from './lab.module.css'

export const metadata = createPageMetadata({
  title: 'Lab',
  description:
    'Interactive experiments on how language models actually work — playgrounds you can poke, not diagrams you squint at.',
  path: '/lab',
})

type Experiment = {
  slug: string
  index: string
  title: string
  hook: string
  tag: string
  date: string
}

const experiments: Experiment[] = [
  {
    slug: '/lab/sampling',
    index: '001',
    title: 'how sampling works',
    hook: 'temperature, top-k, and top-p on a live token distribution — feel the dice reshape before the roll.',
    tag: 'interactive',
    date: '2026-07',
  },
]

export default function Page() {
  return (
    <section className={styles.lab}>
      <p className={styles.kicker}>the lab</p>
      <h1 className={styles.title}>
        Experiments in <em>how models actually work.</em>
      </h1>
      <p className={styles.intro}>
        Interactive notes from taking model internals apart — the mechanisms
        behind the magic, rebuilt small enough to hold in your hands and turn
        over. Less explaining, more poking. Each one runs entirely in your
        browser.
      </p>

      <ol className={styles.grid}>
        {experiments.map((exp) => (
          <li key={exp.slug} className={styles.item}>
            <Link href={exp.slug} className={styles.card}>
              <div className={styles.cardHead}>
                <span className={styles.cardIndex}>{exp.index}</span>
                <span className={styles.cardTag}>{exp.tag}</span>
              </div>
              <h2 className={styles.cardTitle}>{exp.title}</h2>
              <p className={styles.cardHook}>{exp.hook}</p>
              <div className={styles.cardMeta}>
                <time dateTime={exp.date}>{exp.date}</time>
                <span className={styles.cardOpen} aria-hidden="true">
                  open →
                </span>
              </div>
            </Link>
          </li>
        ))}

        <li
          className={`${styles.item} ${styles.placeholder}`}
          aria-hidden="true"
        >
          <div className={styles.placeholderInner}>
            <span className={styles.placeholderIndex}>002</span>
            <p className={styles.placeholderNote}>
              more experiments in the notebook margin — attention maps,
              tokenizers, embeddings…
            </p>
          </div>
        </li>
      </ol>
    </section>
  )
}
