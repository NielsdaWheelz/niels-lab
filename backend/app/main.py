from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router

app = FastAPI(title="Niels Lab API")

app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    "http://localhost:3000",
    # "*",
  ],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/health")
def health():
  return {"status": "ok"}