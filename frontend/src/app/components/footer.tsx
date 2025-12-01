export default function Footer() {
  return (
    <footer>
      <p>
        <a href="https://github.com/NielsdaWheelz" target="_blank" rel="noopener noreferrer">
          github
        </a>
        {' · '}
        <a href="https://x.com/the_powertool" target="_blank" rel="noopener noreferrer">
          twitter
        </a>
        {' · '}
        <a href="https://www.linkedin.com/in/nielseriknandal/" target="_blank" rel="noopener noreferrer">
          linkedin
        </a>
      </p>
      <p style={{ marginTop: '0.5rem' }}>
        © {new Date().getFullYear()}
      </p>
    </footer>
  )
}
