import { ProjectsList } from '@/app/components/projects'
import { Hero } from '@/app/components/Hero'
import { DrawHeading } from '@/app/components/DrawHeading'
import Link from 'next/link'

export default function Page() {
  return (
    <section>
      <Hero />
      
      <DrawHeading as="h2" underlineColor="sage" delay={2200}>
        projects
      </DrawHeading>
      <ProjectsList limit={5} />
      <div style={{ marginTop: '1rem', marginBottom: '2rem' }}>
        <Link href="/projects" className="view-all-link" style={{ fontSize: '0.95em' }}>
          view all projects →
        </Link>
      </div>
      
      <DrawHeading as="h2" underlineColor="terracotta" delay={500}>
        skills
      </DrawHeading>
      <div style={{ marginBottom: '2rem' }}>
        <p style={{ marginBottom: '0.75rem', fontSize: '0.95em' }}>
          <strong>Languages:</strong> Python, TypeScript, SQL
        </p>
        <p style={{ marginBottom: '0.75rem', fontSize: '0.95em' }}>
          <strong>Backend:</strong> FastAPI, SQLAlchemy, PostgreSQL, Redis/Celery
        </p>
        <p style={{ marginBottom: '0.75rem', fontSize: '0.95em' }}>
          <strong>AI/ML:</strong> OpenAI API, embeddings (pgvector), LLM orchestration
        </p>
        <p style={{ marginBottom: '0.75rem', fontSize: '0.95em' }}>
          <strong>Frontend:</strong> React, Next.js, TypeScript
        </p>
        <p style={{ fontSize: '0.95em' }}>
          <strong>Tools:</strong> Docker, Git, pytest
        </p>
      </div>
      
      <DrawHeading as="h2" underlineColor="sage" delay={500}>
        get in touch
      </DrawHeading>
      <div style={{ marginBottom: '2rem' }}>
        <p style={{ marginBottom: '0.75rem', fontSize: '0.95em' }}>
          Looking to hire? I'm currently available for full-time opportunities.
        </p>
        <p style={{ fontSize: '0.95em' }}>
          <Link href="/cv">view my CV</Link>
          {' · '}
          <a href="https://www.linkedin.com/in/nielseriknandal/" target="_blank" rel="noopener noreferrer">
            linkedin
          </a>
          {' · '}
          <a href="https://github.com/NielsdaWheelz" target="_blank" rel="noopener noreferrer">
            github
          </a>
        </p>
      </div>
    </section>
  )
}
