import Link from 'next/link'
import { DrawHeading } from '@/app/components/DrawHeading'
import { ProjectsList } from '@/app/components/projects'
import { getWritingPosts, formatDate } from '@/app/writing/utils'
import { githubUrl, linkedinUrl } from '@/app/site'

export default function Page() {
  const writingPosts = getWritingPosts()
    .sort((a, b) => {
      if (new Date(a.metadata.publishedAt) > new Date(b.metadata.publishedAt)) {
        return -1
      }
      return 1
    })
    .slice(0, 3)

  return (
    <section className="home-page">
      <section className="home-hero">
        <p className="home-kicker">Niels Erik Nandal · engineer’s notebook</p>
        <h1 className="home-title">
          Engineer building <em>AI products</em>, systems, and{' '}
          <span className="circle-annotation">
            readable
            <svg viewBox="0 0 200 60" aria-hidden="true" focusable="false">
              <path d="M 14 30 C 8 10, 60 4, 102 6 C 152 8, 196 14, 193 30 C 190 48, 140 56, 96 55 C 52 54, 4 50, 12 26" />
            </svg>
          </span>{' '}
          software.
        </h1>
        <p className="home-summary">
          I work across Python, TypeScript, and ML-heavy product surfaces. I
          care about deterministic backends, sharp product loops, and code a
          maintainer can understand in one pass.
        </p>
        <p className="home-links">
          <Link href="/cv">view CV</Link>
          {' · '}
          <a href={githubUrl} target="_blank" rel="noopener noreferrer">
            github
          </a>
          {' · '}
          <a href={linkedinUrl} target="_blank" rel="noopener noreferrer">
            linkedin
          </a>
        </p>
        <p>
          <span className="hero-stamp">open to full-time roles</span>
        </p>
        <p className="hero-margin-note">
          <span className="margin-note">
            psst — press ctrl+k. this site has a real terminal ↘
          </span>
        </p>
      </section>

      <DrawHeading as="h2" underlineColor="sage" delay={2200}>
        featured projects
      </DrawHeading>
      <p className="section-intro">
        Selected work that best reflects how I think, what I ship, and the level
        of technical depth I want to be judged on.
      </p>
      <ProjectsList
        slugs={['niels-gpt', 'nexus', 'factory-simulator', 'suno-demo']}
      />
      <p className="section-link">
        <Link href="/projects" className="view-all-link">
          view all projects →
        </Link>
      </p>

      <DrawHeading as="h2" underlineColor="terracotta" delay={500}>
        writing
      </DrawHeading>
      <p className="section-intro">
        Technical notes and build writeups. Short, direct, and focused on the
        work.
      </p>
      <ul className="writing-list">
        {writingPosts.map((post) => (
          <li key={post.slug} className="writing-item">
            <Link href={`/writing/${post.slug}`} className="writing-title">
              {post.metadata.title}
            </Link>
            <p className="writing-meta">
              {formatDate(post.metadata.publishedAt)}
            </p>
            <p className="writing-summary">{post.metadata.summary}</p>
          </li>
        ))}
      </ul>
      <p className="section-link">
        <Link href="/writing" className="view-all-link">
          view all writing →
        </Link>
      </p>
    </section>
  )
}
