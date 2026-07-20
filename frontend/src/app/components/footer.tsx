import { githubUrl, linkedinUrl, xUrl } from '@/app/site'

export default function Footer() {
  return (
    <footer>
      <p>
        <a href={githubUrl} target="_blank" rel="noopener noreferrer">
          github
        </a>
        {' · '}
        <a href={linkedinUrl} target="_blank" rel="noopener noreferrer">
          linkedin
        </a>
        {' · '}
        <a href={xUrl} target="_blank" rel="noopener noreferrer">
          x
        </a>
      </p>
      <p style={{ marginTop: '0.5rem' }}>
        <span className="margin-note">
          drawn in ink, typeset in mono · terminal: ctrl+k
        </span>
      </p>
      <p style={{ marginTop: '0.5rem' }}>© {new Date().getFullYear()}</p>
    </footer>
  )
}
