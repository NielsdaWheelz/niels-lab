from fastapi import FastAPI
from .routes import router

app = FastAPI(title="Niels Lab API")

app.include_router(router, prefix="/api")

@app.get("/health")
def health():
  return {"status": "ok"}