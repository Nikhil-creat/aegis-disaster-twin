# Aegis — Agentic Disaster Response Digital Twin

**Designed & Developed by 
# [NIKHIL CHARY SRIRAMOJU](https://github.com/Nikhil-creat)**
[GitHub](https://github.com/Nikhil-creat) 
[LinkedIn](https://in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a) 
sriramojunikhil66@gmail.com

**Live demo:** _add your GitHub Pages link here after deploying (see below)_

Aegis is a multi-service system that turns raw disaster-zone imagery and
sensor telemetry into a live 3D digital twin, then coordinates an
autonomous multi-agent response team to triage, allocate resources, and
draft protocol-grounded alerts — without a human manually reading every
report first.

## Why this exists

During floods, earthquakes, and large-scale fires, the bottleneck usually
isn't a lack of data — it's the time it takes a human to fuse imagery,
sensor feeds, and response protocols into a decision. Aegis compresses
that loop: CNN-based damage assessment feeds a multi-agent orchestrator
that reasons over the situation and is grounded, at every step, in real
emergency-management protocol text via retrieval-augmented generation.

## Architecture

```
                    ┌─────────────────────┐
  drone / satellite │  Ingestion Gateway  │  IoT sensors (water level,
  imagery tiles ───▶│     (FastAPI)       │◀── seismic, structural strain)
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌──────────────────┐
        │  CNN Inference   │        │   RAG Service     │
        │  (U-Net, damage  │        │  (FAISS + MiniLM, │
        │  segmentation)   │        │  NDMA/FEMA SOPs)   │
        └────────┬─────────┘        └─────────┬──────────┘
                 │                             │
                 ▼                             │
        ┌───────────────────────────────────────┐
        │       Agent Orchestrator (LangGraph)   │
        │  TriageAgent → ResourceAgent → CommsAgent │
        └───────────────────┬─────────────────────┘
                             ▼
                 ┌───────────────────────┐
                 │  3D Digital Twin (UI)  │
                 │  Three.js + live state │
                 └───────────────────────┘
```

## Services

| Service | Path | Responsibility |
|---|---|---|
| `ingestion-gateway` | `services/ingestion-gateway` | Normalizes imagery + sensor payloads; simulated telemetry endpoint for offline demo |
| `cnn-inference` | `services/cnn-inference` | U-Net damage segmentation → severity class, confidence, flood extent per zone |
| `rag-service` | `services/rag-service` | FAISS vector index over real emergency SOP documents; grounds every agent decision |
| `agent-orchestrator` | `services/agent-orchestrator` | LangGraph state machine: Triage → Resource Allocation → Comms drafting |
| `frontend` | `frontend/` | Three.js 3D digital twin + live agent log + alert feed (this is what's deployed to GitHub Pages) |

## Running the full stack locally

```bash
docker-compose up --build
```

This starts all four backend services on ports `8000`–`8003`. Open
`frontend/index.html` in a browser, or serve it with any static file
server, to interact with the twin against live services once you wire
the frontend's fetch calls to `http://localhost:8000..8003` (the
GitHub Pages build runs in a self-contained simulation mode instead,
since Pages can't host the Python services).

## The GitHub Pages demo vs. the full system

GitHub Pages only serves static files, so it cannot run FastAPI, PyTorch,
or FAISS. The deployed demo (`frontend/index.html`) runs the **same
triage → allocation → comms logic** as `agents.py`, client-side, against
a seeded synthetic dataset, so the interaction model, the decision
logic, and the protocol citations are real — only the network calls are
replaced with in-browser computation. The `services/` directory is the
actual backend you'd deploy behind it in a real environment (e.g. on
Render, Fly.io, or a Kubernetes cluster), and every service exposes a
`/health` endpoint plus OpenAPI docs at `/docs` when running.

## Reliability & performance engineering

This isn't a script that only works on the happy path. Each concern below
maps to real code in `services/`, not just words in this file:

| Concern | Implementation |
|---|---|
| **Transient failures** | `agent-orchestrator/rag_client.py` retries with exponential backoff + jitter (bounded at 3 attempts) before giving up |
| **Cascading failure** | A circuit breaker trips after 5 consecutive RAG failures, short-circuits further calls for a 30s cooldown, then half-opens to test recovery — the orchestrator never hangs waiting on a degraded dependency |
| **Graceful degradation** | If RAG is unreachable, agents still run — alerts are produced without a cited protocol source rather than failing the whole response cycle |
| **Liveness vs readiness** | Every service exposes both `/health` (process is up) and `/ready` (dependencies are actually usable — model loaded, FAISS index built, circuit breaker state) so an orchestrator only routes traffic to instances that can really serve |
| **Observability** | Structured request logging middleware records method, path, status, and latency per request — the basis for the p95/latency panel in the frontend |
| **Container resilience** | `docker-compose.yml` defines healthchecks, `restart: unless-stopped`, and resource limits per service, and `agent-orchestrator` waits on dependency healthchecks (not just container start) before accepting traffic |
| **Resource-constrained inference** | The CNN service runs a lightweight custom U-Net (not a full ResNet backbone) so it stays responsive on modest hardware, with a random-init fallback so the service boots cleanly even without trained weights |

The frontend's **SYSTEM RELIABILITY** panel visualizes this pattern — the
GitHub Pages build simulates the numbers since it can't run the Python
services, but the retry/circuit-breaker code it's illustrating is real
and lives in `services/agent-orchestrator/rag_client.py`.

## Tech stack

- **Computer vision:** PyTorch, U-Net-style segmentation
- **Agentic AI:** LangGraph multi-agent state graph, LangChain core
- **RAG:** FAISS + `sentence-transformers/all-MiniLM-L6-v2`
- **Backend:** FastAPI, Pydantic, Docker Compose
- **Frontend / digital twin:** Three.js, vanilla JS, no build step

## Repository structure

```
aegis-disaster-twin/
├── frontend/
│   └── index.html              # 3D digital twin (GitHub Pages entry point)
├── services/
│   ├── ingestion-gateway/
│   ├── cnn-inference/
│   ├── rag-service/
│   └── agent-orchestrator/
├── data/sample-protocols/       # emergency SOP text used for RAG grounding
├── docker-compose.yml
└── README.md
```

## Roadmap

- Swap synthetic telemetry for a real MQTT/Kafka feed
- Fine-tune the CNN on the full xBD + FloodNet datasets
- Add a fourth agent for post-event resource reconciliation
- WebSocket layer so the twin updates from live backend state, not replay

## Author

**NIKHIL CHARY SRIRAMOJU** 
— BTech CSE (Final Year)
GitHub: [Nikhil-creat](https://github.com/Nikhil-creat) · LinkedIn: [nikhil-chary-sriramoju](https://in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a) · Email: sriramojunikhil66@gmail.com
