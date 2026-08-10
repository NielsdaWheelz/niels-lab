export const entries = [
  {
    title: 'Senior Software Engineer – Solid',
    date: 'March 2026 – present',
    category: 'experience' as const,
    bullets: [
      'Build infrastructure that lets an agent hold work beyond a single chat session: persistent memory, machines, accounts, communications, payments.',
      'Ship production backend services: typed contracts at the boundaries, deterministic cores, migrations that roll back in one revision.',
    ],
  },
  {
    title: 'Fractal Tech – Software Engineering Fellowship',
    subtitle: 'Full-time, six-day weeks, 13-hour days',
    date: 'October 2025 – December 2025',
    category: 'education' as const,
  },
  {
    title: 'Nexus',
    date: 'November 2025',
    category: 'project' as const,
    bullets: [
      'Built a reading system that ingests PDFs, EPUBs, and HTML into deterministic canonical text, chunked and embedded into Postgres/pgvector for semantic search.',
      'Anchored highlights and annotations as byte-offset ranges into canonical text.',
      'Enforced per-user access control on every search and document read.',
      'Served it from FastAPI, SQLAlchemy, and Alembic with Celery workers, fronted by Next.js with a TypeScript API client generated from the OpenAPI schema.',
    ],
  },
  {
    title: 'Factory Simulator',
    date: 'November 2025',
    category: 'project' as const,
    bullets: [
      'Built LLM agents that turn free-text factory descriptions into typed configs for a deterministic EDD job-shop scheduler.',
      'Validated model output in one function: OpenAI JSON mode parsed into Pydantic schemas.',
      'Scored every parse against machine and job IDs regex-extracted from the source text.',
      'Wrote 419 backend tests; the suite mocks the model boundary and needs no API key.',
    ],
  },
  {
    title: 'Intern – SupplyCo',
    date: 'October – November 2025',
    category: 'experience' as const,
    bullets: [
      'Refactored production Dagster pipelines from ad-hoc Supabase/SQL scripts into typed SQLModel DAOs, pure transformation layers, and an orchestrator, removing pandas outright.',
      'Replaced manual schema management with an Alembic/SQLAlchemy migration system — baseline revisions, whitelists, schema-diff tooling, autogenerate gating.',
      'Replaced mock-based tests with an ephemeral Alembic-stamped Postgres harness, ending silent ORM/DB drift.',
    ],
  },
  {
    title: 'Product Manager – Mind Brain Behaviour Hive',
    date: 'December 2018 – June 2020',
    category: 'experience' as const,
    bullets: [
      'Translated cognitive-neuroscience research into shipped browser, mobile, AR, and VR applications: specifications, experiment design, and delivery with the engineers.',
    ],
  },
  {
    title: 'R&D Consultant – Mind Brain Behaviour Hive',
    date: 'October 2015 – March 2016, July 2018 – December 2018',
    category: 'experience' as const,
    bullets: [
      'Built detection algorithms over physiological, text and speech, and facial and gestural signals, turning raw behavioral data into features, metrics, and visualizations of emotional and cognitive state.',
      'Prototyped a VR stress-regulation game and a Bluetooth breathing and heart-rate monitor used in research workflows.',
    ],
  },
  {
    title:
      'Research Assistant – University of Toronto: Rotman School of Management',
    date: 'July 2015 – March 2016',
    category: 'experience' as const,
    bullets: [
      'Designed and ran behavioral experiments on multi-alternative choice: attraction, compromise, and dominance effects.',
      'Designed a habituation experiment with Dr. Colin Camerer on how repeated exposure alters valuation and choice.',
    ],
  },
  {
    title: 'McGill University',
    subtitle: 'BSc. Molecular Biology',
    date: '2010 – 2014',
    category: 'education' as const,
  },
]

export const skills = {
  Languages: ['Python', 'TypeScript', 'SQL'],
  Backend: [
    'FastAPI',
    'Alembic',
    'SQLAlchemy',
    'SQLModel',
    'PostgreSQL',
    'Redis',
    'Celery',
    'Express',
  ],
  'AI/ML': ['PyTorch', 'scikit-learn', 'Pandas', 'NumPy', 'RAG', 'LangGraph'],
  Frontend: ['React', 'Next.js', 'TanStack Query', 'Tailwind', 'Vite'],
  Tools: ['Docker', 'Git', 'JSON', 'Linux', 'Bash', 'Make'],
}
