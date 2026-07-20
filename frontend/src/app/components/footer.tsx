import { githubUrl, linkedinUrl, xUrl } from '@/app/site'

export default function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-signoff">
        <span className="footer-mark" aria-hidden="true">
          N/E
        </span>
        <p>
          Ambitious software.
          <br />
          Legible systems.
        </p>
      </div>
      <div className="footer-meta">
        <p className="footer-links">
          <a href={githubUrl} target="_blank" rel="me noopener noreferrer">
            github
          </a>
          {' · '}
          <a href={linkedinUrl} target="_blank" rel="me noopener noreferrer">
            linkedin
          </a>
          {' · '}
          <a href={xUrl} target="_blank" rel="me noopener noreferrer">
            x
          </a>
        </p>
        <p className="footer-note">paper, ink, code · terminal: ctrl+k</p>
        <p>© {new Date().getFullYear()} Niels Erik Nandal</p>
      </div>
    </footer>
  )
}
