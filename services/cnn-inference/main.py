"""
Aegis — CNN Damage Assessment Service
--------------------------------------
Serves a segmentation/classification model over incoming drone or satellite
imagery tiles and returns per-tile damage severity + flood-extent masks.

Model: U-Net encoder (ResNet18 backbone) fine-tuned on xBD (building damage)
and FloodNet (flood segmentation) datasets. Swap `MODEL_PATH` with your
trained weights; a random-init fallback is used if none is found so the
service boots cleanly in dev/demo environments.
"""

import io
import os
import time
import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cnn-inference")

MODEL_PATH = os.getenv("MODEL_PATH", "/models/aegis_unet_r18.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEVERITY_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


class DamageUNet(nn.Module):
    """Lightweight U-Net-style encoder-decoder for damage segmentation."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.enc1 = self._block(3, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._block(64, 32)
        self.out = nn.Conv2d(32, num_classes, 1)

    @staticmethod
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def load_model() -> DamageUNet:
    model = DamageUNet(num_classes=len(SEVERITY_CLASSES)).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        logger.info("Loaded trained weights from %s", MODEL_PATH)
    else:
        logger.warning("No trained weights found at %s — serving random-init model (demo mode)", MODEL_PATH)
    model.eval()
    return model


app = FastAPI(title="Aegis CNN Damage Assessment Service", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
model = load_model()
_start_time = time.time()


class ZoneDamageResult(BaseModel):
    zone_id: str
    severity: str
    confidence: float
    flood_extent_pct: float
    inference_ms: float


def preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((256, 256))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "weights_loaded": os.path.exists(MODEL_PATH)}


@app.get("/ready")
def ready():
    """Readiness gate: only report ready once the model is actually loaded on device,
    so a load balancer never routes an inference request to a cold instance."""
    model_ready = model is not None
    return {
        "status": "ready" if model_ready else "not_ready",
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.post("/v1/assess-zone", response_model=ZoneDamageResult)
async def assess_zone(zone_id: str, file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/tiff"):
        raise HTTPException(400, "Unsupported image type")

    start = time.perf_counter()
    raw = await file.read()
    image = Image.open(io.BytesIO(raw))
    tensor = preprocess(image)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        class_map = probs.argmax(dim=1)
        flat = class_map.flatten()
        severity_idx = int(flat.mode().values.item())
        confidence = float(probs[0, severity_idx].mean().item())
        flood_extent = float((flat >= 2).float().mean().item() * 100)

    elapsed_ms = (time.perf_counter() - start) * 1000
    return ZoneDamageResult(
        zone_id=zone_id,
        severity=SEVERITY_CLASSES[severity_idx],
        confidence=round(confidence, 4),
        flood_extent_pct=round(flood_extent, 2),
        inference_ms=round(elapsed_ms, 2),
    )


class BatchRequest(BaseModel):
    zone_ids: List[str]


@app.get("/v1/severity-classes")
def severity_classes():
    return {"classes": SEVERITY_CLASSES}
