from sqlmodel import SQLModel, create_engine, Session
from .config import settings
from .models import Post, Project

engine = create_engine(settings.DATABASE_URL, echo=True)

def get_session():
  with Session(engine) as session:
    yield session

# temporary to init db
def init_db():
  SQLModel.metadata.create_all(engine)