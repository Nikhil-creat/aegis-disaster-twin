import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

from agents import run_response_cycle, ZoneAssessment
from rag_client import _breaker as rag_breaker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("agent-orchestrator")

app = FastAPI(title="Aegis Agent Orchestrator", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_start_time = time.time()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Structured request logging — every call's latency is recorded so p95/p99
    can be derived without a separate APM agent in the demo environment."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %d (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


class ResponseCycleRequest(BaseModel):
    zones: List[ZoneAssessment]
    available_resources: Dict[str, int]


@app.get("/health")
def health():
    """Liveness probe — process is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness probe — reports downstream dependency health so an
    orchestrator (k8s, ECS) can pull this instance from rotation if its
    RAG dependency's circuit breaker has tripped."""
    return {
        "status": "ready" if rag_breaker.state.value != "open" else "degraded",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "rag_circuit_breaker": rag_breaker.state.value,
    }


@app.post("/v1/run-cycle")
def run_cycle(req: ResponseCycleRequest):
    result = run_response_cycle(req.zones, req.available_resources)
    return {
        "priority_order": result["priority_order"],
        "allocations": result["allocations"],
        "remaining_resources": result["available_resources"],
        "alerts": result["alerts"],
        "log": result["log"],
    }
