"""
Aegis — Ingestion Gateway
---------------------------
Single entry point for incoming drone/satellite imagery tiles and IoT sensor
telemetry (water level, seismic, structural strain). Normalizes payloads and
fans them out to the CNN inference service and the orchestrator.

In production this would subscribe to an MQTT broker / Kafka topic fed by
field hardware. For this deployment it exposes a REST endpoint plus a
simulated telemetry generator used by the demo frontend.
"""

import random
import time
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Aegis Ingestion Gateway", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_start_time = time.time()

ZONE_IDS = [f"Z-{i:02d}" for i in range(1, 13)]


class SensorReading(BaseModel):
    zone_id: str
    water_level_cm: float
    seismic_activity: float
    structural_strain_pct: float
    timestamp: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"status": "ready", "uptime_seconds": round(time.time() - _start_time, 1)}


@app.get("/v1/simulate-telemetry")
def simulate_telemetry() -> Dict[str, list]:
    """Generates plausible sensor readings for demo / offline mode."""
    readings = []
    for zone_id in ZONE_IDS:
        readings.append(
            SensorReading(
                zone_id=zone_id,
                water_level_cm=round(random.uniform(5, 180), 1),
                seismic_activity=round(random.uniform(0, 6.5), 2),
                structural_strain_pct=round(random.uniform(0, 95), 1),
                timestamp=time.time(),
            ).dict()
        )
    return {"readings": readings}
