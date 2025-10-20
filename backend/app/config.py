from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
  DATABASE_URL: str = Field(
    default="postgresql+psycopg://fractal:fractal@localhost:5432/postgres"
  )

  class Config:
    env_file = ".env"

settings = Settings()