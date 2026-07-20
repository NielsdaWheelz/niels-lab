import Link from 'next/link'
import { ProjectsList } from '@/app/components/projects'
import { getWritingPosts, formatDate } from '@/app/writing/utils'
import { githubUrl, linkedinUrl } from '@/app/site'
import { SystemMap } from '@/app/components/SystemMap'
import { TerminalLauncher } from '@/app/components/TerminalLauncher'

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
    <div className="home-page">
      <section className="home-hero" aria-labelledby="home-title">
        <div className="home-hero-copy">
          <p className="home-kicker">
            <span className="status-dot" aria-hidden="true" />
            Niels Erik Nandal · AI systems engineer
          </p>
          <h1 id="home-title" className="home-title">
            Build the <em>magic.</em>
            <br />
            Keep the{' '}
            <span className="circle-annotation">
              receipts.
              <svg viewBox="0 0 250 76" aria-hidden="true" focusable="false">
                <path d="M 15 38 C 8 12, 72 4, 130 7 C 194 9, 242 18, 238 39 C 234 62, 174 71, 116 68 C 58 66, 5 61, 13 32" />
              </svg>
            </span>
          </h1>
          <p className="home-summary">
            I build ambitious AI products that stay legible underneath—model
            internals, deterministic backends, sharp interfaces, and the product
            loops between them.
          </p>
          <div className="hero-actions">
            <Link
              href="#selected-work"
              className="hero-action hero-action-primary"
            >
              inspect selected work <span aria-hidden="true">↓</span>
            </Link>
            <TerminalLauncher />
          </div>
          <p className="hero-availability">
            <span aria-hidden="true" />
            Open to full-time AI / software engineering roles
          </p>
          <p className="margin-note hero-margin-note">
            new in the notebook →{' '}
            <Link href="/lab">go poke a token sampler in the lab</Link>
          </p>
        </div>
        <SystemMap />
      </section>

      <dl className="proof-strip" aria-label="Current engineering profile">
        <div>
          <dt>current</dt>
          <dd>Fractal Tech · software engineering fellow</dd>
        </div>
        <div>
          <dt>range</dt>
          <dd>model internals → product interfaces</dd>
        </div>
        <div>
          <dt>toolkit</dt>
          <dd>Python · TypeScript · PostgreSQL</dd>
        </div>
      </dl>

      <section id="selected-work" className="home-section home-work-section">
        <header className="section-header">
          <p className="section-index">01 / selected systems</p>
          <h2>Proof, not promises.</h2>
          <p>
            Each project is a working argument: ambitious behavior belongs above
            explicit contracts, deterministic cores, and interfaces that show
            their work.
          </p>
        </header>
        <ProjectsList
          variant="showcase"
          slugs={['nexus', 'niels-gpt', 'factory-simulator', 'suno-demo']}
        />
        <p className="section-link">
          <Link href="/projects" className="view-all-link">
            open the full project index <span aria-hidden="true">↗</span>
          </Link>
        </p>
      </section>

      <section className="home-section principles-section">
        <header className="section-header">
          <p className="section-index">02 / operating principles</p>
          <h2>How I make hard systems hold together.</h2>
        </header>
        <ol className="principles-grid">
          <li>
            <span>01</span>
            <h3>Magic above. Rigor below.</h3>
            <p>
              Models interpret and generate; typed contracts and deterministic
              systems protect the truth.
            </p>
          </li>
          <li>
            <span>02</span>
            <h3>Reveal the machine.</h3>
            <p>
              Good interfaces expose state, lineage, and tradeoffs so a human
              can steer with confidence.
            </p>
          </li>
          <li>
            <span>03</span>
            <h3>Readable is a feature.</h3>
            <p>
              Clear seams, traces, and tests make speed compound instead of
              turning into debt.
            </p>
          </li>
        </ol>
      </section>

      <section className="home-section writing-section">
        <header className="section-header section-header-split">
          <div>
            <p className="section-index">03 / field notes</p>
            <h2>Thinking in public.</h2>
          </div>
          <Link href="/writing" className="view-all-link">
            all writing <span aria-hidden="true">↗</span>
          </Link>
        </header>
        <ol className="home-writing-list">
          {writingPosts.map((post, index) => (
            <li key={post.slug}>
              <span className="writing-number">
                {String(index + 1).padStart(2, '0')}
              </span>
              <div>
                <Link href={`/writing/${post.slug}`} className="writing-title">
                  {post.metadata.title}
                </Link>
                <p className="writing-summary">{post.metadata.summary}</p>
              </div>
              <time dateTime={post.metadata.publishedAt}>
                {formatDate(post.metadata.publishedAt)}
              </time>
            </li>
          ))}
        </ol>
      </section>

      <section className="home-cta" aria-labelledby="home-cta-title">
        <p className="section-index">04 / compare notes</p>
        <h2 id="home-cta-title">
          Hard problem somewhere between a model and a real user?
        </h2>
        <p>I like that territory.</p>
        <div className="home-cta-links">
          <a
            href="mailto:niels.erik.nandal@gmail.com"
            className="hero-action hero-action-primary"
          >
            email me <span aria-hidden="true">↗</span>
          </a>
          <Link href="/cv" className="hero-action hero-action-secondary">
            read the CV
          </Link>
          <a
            href={githubUrl}
            target="_blank"
            rel="me noopener noreferrer"
            className="text-link"
          >
            GitHub
          </a>
          <a
            href={linkedinUrl}
            target="_blank"
            rel="me noopener noreferrer"
            className="text-link"
          >
            LinkedIn
          </a>
        </div>
      </section>
    </div>
  )
}
