import './print.css'
import { Fragment } from 'react'
import { PageTitle } from '@/app/components/PageTitle'
import { JsonLd } from '@/app/components/StructuredData'
import {
  createPageMetadata,
  getCanonicalUrl,
  getPersonSchemaId,
  getWebsiteSchemaId,
  githubUrl,
  linkedinUrl,
  siteName,
} from '@/app/site'
import { entries, skills } from './data'
import styles from './cv.module.css'

const description =
  'Roles, projects, education, and tools. Niels-Erik Nandal is a senior software engineer at Solid in San Francisco.'

export const metadata = createPageMetadata({
  title: 'CV',
  description,
  path: '/cv',
})

// The three headings the entries sort under; order is the reading order.
const sections = [
  { heading: 'Experience', category: 'experience' },
  { heading: 'Projects', category: 'project' },
  { heading: 'Education', category: 'education' },
] as const

export default function CVPage() {
  const cvUrl = getCanonicalUrl('/cv')

  return (
    <>
      <JsonLd
        data={{
          '@context': 'https://schema.org',
          '@type': 'ProfilePage',
          '@id': `${cvUrl}#profile`,
          url: cvUrl,
          name: `CV — ${siteName}`,
          description,
          inLanguage: 'en-US',
          isPartOf: { '@id': getWebsiteSchemaId() },
          mainEntity: { '@id': getPersonSchemaId() },
          about: { '@id': getPersonSchemaId() },
        }}
      />
      <article>
        <PageTitle>cv</PageTitle>
        <p className={styles.bio}>
          Niels-Erik Nandal is a senior software engineer at Solid in San
          Francisco.
        </p>
        <p className={`chrome ${styles.contact}`}>
          <a href="mailto:niels.erik.nandal@gmail.com">
            niels.erik.nandal@gmail.com
          </a>
          {' · '}
          <a href={githubUrl}>github.com/NielsdaWheelz</a>
          {' · '}
          <a href={linkedinUrl}>linkedin.com/in/nielseriknandal</a>
        </p>

        {sections.map((section) => (
          <section key={section.category} className={styles.section}>
            <h2 className={styles.sectionHeading}>{section.heading}</h2>
            {entries
              .filter((entry) => entry.category === section.category)
              .map((entry) => (
                <div key={entry.title} className={styles.item}>
                  <div className={styles.row}>
                    <h3 className={styles.role}>{entry.title}</h3>
                    <span className="date">{entry.date}</span>
                  </div>
                  {'subtitle' in entry && entry.subtitle ? (
                    <p className={styles.subtitle}>{entry.subtitle}</p>
                  ) : null}
                  {'bullets' in entry && entry.bullets ? (
                    <ul className="entries">
                      {entry.bullets.map((bullet) => (
                        <li key={bullet} className="entry">
                          {bullet}
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              ))}
          </section>
        ))}

        <section className={styles.section}>
          <h2 className={styles.sectionHeading}>Skills</h2>
          <dl className={styles.skills}>
            {Object.entries(skills).map(([group, tags]) => (
              <Fragment key={group}>
                <dt>{group}</dt>
                <dd>{tags.join(', ')}</dd>
              </Fragment>
            ))}
          </dl>
        </section>
      </article>
    </>
  )
}
