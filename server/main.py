"""
Tile Visualizer — FastAPI Server

Run:  uvicorn server.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .routes import routers
from .services import get_model

app = FastAPI(
    title="Tile Visualizer API",
    description="Apply tiles to room surfaces using Detectron2 + SAM segmentation",
    version="1.0.0",
)

# CORS — allow any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved outputs as static files
outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
outputs_dir.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

# Routes
for r in routers:
    app.include_router(r)


@app.on_event("startup")
def preload_model():
    """Load the segmentation model once at startup."""
    get_model()


@app.get("/health")
def health():
    return {"status": "ok"}
