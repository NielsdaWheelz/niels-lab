from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class PostBase(SQLModel):
  title: str
  slug: str = Field(index=True)
  content: str

class Post(PostBase, table=True):
  id: Optional[int] = Field(default=None, primary_key=True)
  created_at: datetime = Field(default_factory=datetime.now, index=True)
  updated_at: datetime = Field(default_factory=datetime.now)

class ProjectBase(SQLModel):
    name: str
    slug: str = Field(index=True, unique=True)
    description: str = ""
    repo_url: Optional[str] = None
    live_url: Optional[str] = None

class Project(ProjectBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now, index=True)
    updated_at: datetime = Field(default_factory=datetime.now)