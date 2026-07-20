import { entries, skills } from './data'

const CATEGORY_COLORS = {
  experience: 'var(--color-terracotta)',
  project: 'var(--color-sage)',
  education: 'var(--color-gold)',
}

export default function Timeline() {
  return (
    <>
      <ol
        className="timeline"
        aria-label="Professional experience, projects, and education"
        style={{ listStyle: 'none' }}
      >
        {entries.map((entry) => (
          <li
            key={`${entry.date}-${entry.title}`}
            className="timeline-entry visible"
          >
            <span
              className="timeline-dot visible"
              style={{
                borderColor: CATEGORY_COLORS[entry.category],
                backgroundColor: CATEGORY_COLORS[entry.category],
              }}
              aria-hidden="true"
            />
            <article className="timeline-card">
              <h2 className="timeline-title">{entry.title}</h2>
              {'subtitle' in entry && entry.subtitle && (
                <p className="timeline-subtitle">{entry.subtitle}</p>
              )}
              <p className="timeline-date">{entry.date}</p>
              {'bullets' in entry && entry.bullets && (
                <ul className="timeline-bullets">
                  {entry.bullets.map((bullet) => (
                    <li key={bullet}>{bullet}</li>
                  ))}
                </ul>
              )}
            </article>
          </li>
        ))}
      </ol>

      <section className="timeline-skills" aria-labelledby="cv-skills-title">
        <h2
          id="cv-skills-title"
          className="timeline-title"
          style={{ marginBottom: '1rem' }}
        >
          Skills
        </h2>
        {Object.entries(skills).map(([group, tags]) => (
          <section key={group} className="timeline-skill-group">
            <h3 className="timeline-skill-label">{group}</h3>
            <ul
              className="timeline-skill-tags"
              style={{ listStyle: 'none', paddingLeft: 0 }}
            >
              {tags.map((tag) => (
                <li key={tag} className="timeline-skill-tag visible">
                  {tag}
                </li>
              ))}
            </ul>
          </section>
        ))}
      </section>
    </>
  )
}
