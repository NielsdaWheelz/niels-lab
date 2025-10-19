from fastapi import FastAPI

app = FastAPI(title="Niels Lab Backend")

@app.get("/health")
def health():
  return {"status": "ok"}