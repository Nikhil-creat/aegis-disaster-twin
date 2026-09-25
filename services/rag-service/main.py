"""
Aegis — RAG Grounding Service
------------------------------
Retrieves relevant passages from real emergency-management protocol
documents (NDMA / FEMA / WHO mass-casualty guidelines) so agent decisions
cite an actual source instead of hallucinating procedure.

Vector store: FAISS (local, fast, no external dependency).
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (small, CPU-friendly).
"""

import os
import glob
import logging
from typing import List, Dict

import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-service")

DOCS_DIR = os.getenv("PROTOCOL_DOCS_DIR", "/data/sample-protocols")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

import time

app = FastAPI(title="Aegis RAG Grounding Service", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_start_time = time.time()


class RAGIndex:
    def __init__(self, docs_dir: str):
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.chunks: List[Dict] = []
        self.index: faiss.IndexFlatL2 | None = None
        self._build(docs_dir)

    def _build(self, docs_dir: str) -> None:
        paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
        if not paths:
            logger.warning("No protocol documents found in %s — RAG will return empty results", docs_dir)
            return

        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for i in range(0, len(text), 500):
                chunk = text[i : i + 500].strip()
                if chunk:
                    self.chunks.append({"text": chunk, "source": os.path.basename(path)})

        embeddings = self.embedder.encode([c["text"] for c in self.chunks], convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))
        logger.info("Indexed %d chunks from %d documents", len(self.chunks), len(paths))

    def search(self, query: str, k: int = 3) -> List[Dict]:
        if self.index is None or not self.chunks:
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(q_emb, min(k, len(self.chunks)))
        return [
            {**self.chunks[idx], "score": float(distances[0][i])}
            for i, idx in enumerate(indices[0])
            if idx != -1
        ]


rag_index = RAGIndex(DOCS_DIR)


class Query(BaseModel):
    query: str
    k: int = 3


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(rag_index.chunks)}


@app.get("/ready")
def ready():
    """Not ready until the FAISS index has at least one vector — prevents
    the orchestrator from getting empty-context responses during cold start."""
    index_ready = rag_index.index is not None and len(rag_index.chunks) > 0
    return {
        "status": "ready" if index_ready else "not_ready",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "chunks_indexed": len(rag_index.chunks),
    }


@app.post("/v1/retrieve")
def retrieve(q: Query):
    results = rag_index.search(q.query, q.k)
    return {"query": q.query, "results": results}
