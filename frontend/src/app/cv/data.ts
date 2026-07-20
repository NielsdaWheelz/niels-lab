export const entries = [
  {
    title: 'Fractal Tech – Software Engineering Fellowship',
    subtitle: 'Full-time, six-day weeks, 13-hour days',
    date: 'October 2025 – present',
    category: 'education' as const,
  },
  {
    title: 'Nexus',
    date: 'November 2025',
    category: 'project' as const,
    bullets: [
      'Designed and implemented a reading-first knowledge management system: ingesting PDFs/EPUBs/HTML into deterministic canonical text, chunking, and embedding content into Postgres/pgvector for semantic search.',
      'Built a persistent anchoring model where highlights and annotations are stored as byte-offset ranges into canonical text, then remapped on re-ingestion, so annotations survive document updates without breaking.',
      'Architected the service as a modern web stack (FastAPI + SQLAlchemy/Alembic, Redis/Celery workers, Next.js frontend with generated TypeScript API client), with per-user ACL enforced on all search and document access.',
      'Implemented LLM-augmented conversations (thread and message models) as the foundation for a RAG-style "chat with your documents and authors" feature.',
    ],
  },
  {
    title: 'Factory Simulator',
    date: 'November 2025',
    category: 'project' as const,
    bullets: [
      'Built a multi-agent LLM system wrapped around a fully deterministic factory-scheduling simulator—LLM agents handle interpretation/reporting, while the core EDD scheduler and metrics engine remain pure, auditable computation.',
      'Designed a 10-stage orchestration pipeline with strict validation (Pydantic schemas, ID-coverage guarantees, no-hallucination rules) and safe fallbacks, exposing full pipeline traces via a debug API and React UI.',
      'Developed comprehensive tests covering determinism, purity, contract enforcement, agent behavior (LLM-mocked), and end-to-end pipeline invariants.',
    ],
  },
  {
    title: 'Intern – SupplyCo',
    date: 'October – November 2025',
    category: 'experience' as const,
    bullets: [
      'Refactored production Dagster pipelines from ad-hoc Supabase/SQL scripts into a clean SQLModel-based architecture—typed DAOs for all I/O, pure transformation layers, and an orchestrator—eliminating pandas, reducing regressions, and making the workflow deterministic and testable.',
      'Rebuilt the database layer by replacing manual Supabase schema management with a full Alembic/SQLAlchemy migration system—baseline revisions, whitelists, schema diff tooling, and autogenerate gating—giving the team reproducible, version-controlled schema changes for the first time.',
      'Replaced brittle mock-based tests with an ephemeral Postgres test harness (Alembic-stamped), adding granular tests for migration filters and schema scope—ending silent ORM/DB drift and giving the pipeline real integration coverage.',
    ],
  },
  {
    title: 'Product Manager – Mind Brain Behaviour Hive',
    date: 'December 2018 – June 2020',
    category: 'experience' as const,
    bullets: [
      'Translated cognitive-neuroscience research into shipped software products—defining specifications, shaping experiments, and working with engineers to deliver browser, mobile, AR, and VR applications.',
    ],
  },
  {
    title: 'R&D Consultant – Mind Brain Behaviour Hive',
    date: 'October 2015 – March 2016, July 2018 – December 2018',
    category: 'experience' as const,
    bullets: [
      'Developed detection algorithms across multiple signal channels (physiology, text/speech, facial/gestural cues), transforming raw behavioral data into feature sets, metrics, and visualizations used to model emotional and cognitive states.',
      'Prototyped applied systems including a VR stress-regulation game and a Bluetooth breathing/HR monitor integrated into research workflows.',
    ],
  },
  {
    title:
      'Research Assistant – University of Toronto: Rotman School of Management',
    date: 'July 2015 – March 2016',
    category: 'experience' as const,
    bullets: [
      'Designed and ran behavioural experiments on multi-alternative choice (attraction, compromise, dominance effects), analyzing how contextual options shape decision formation.',
      'Developed a habituation experiment in collaboration with Dr. Colin Camerer, focusing on how repeated exposure alters valuation and choice behaviour.',
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
  'AI/ML': [
    'PyTorch',
    'scikit-learn',
    'Pandas',
    'NumPy',
    'RAG',
    'LangGraph',
    'Agent Orchestration',
  ],
  Frontend: ['React', 'Next.js', 'TanStack Query', 'Tailwind', 'Vite'],
  Tools: ['Docker', 'Git', 'JSON', 'Linux', 'Bash', 'Make'],
}
