# -------- SmartBin API (low-memory, Render-friendly) --------
import os

# Keep memory use low on Render Free (512 MB)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2

import torch
torch.set_num_threads(1)
cv2.setNumThreads(0)

from ultralytics import YOLO
import urllib.request, pathlib

# ---------- FastAPI app ----------
app = FastAPI(title="SmartBin API", version="1.0.0")

# CORS: allow your Vercel site and (for now) anything else
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://smartbin-virid.vercel.app",
        "https://*.vercel.app",
        "*",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    max_age=86400,
)

# ---------- Model bootstrap ----------
MODEL_PATH = None
MODEL = None
CLASSES = [
    "paper_cardboard","lvp_plastic_metal","glass_white",
    "glass_brown","glass_green","residual","battery"
]

def ensure_model_path() -> str:
    """Get model path; if SMARTBIN_MODEL is a URL, download to /tmp."""
    global MODEL_PATH
    if MODEL_PATH:
        return MODEL_PATH
    src = os.getenv("SMARTBIN_MODEL", "").strip()
    if not src:
        raise RuntimeError("SMARTBIN_MODEL env missing (local path or https URL)")
    if src.startswith("http"):
        dst = pathlib.Path("/tmp/smartbin/best.pt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            print(f"Downloading model from {src} -> {dst}")
            urllib.request.urlretrieve(src, dst)
        MODEL_PATH = str(dst)
    else:
        MODEL_PATH = src
    return MODEL_PATH

def warm_load():
    """Load the YOLO model once on startup (CPU)."""
    global MODEL
    try:
        mpath = ensure_model_path()
        MODEL = YOLO(mpath)  # CPU only on Render free
        print("✅ YOLO model loaded from", mpath)
    except Exception as e:
        print("❌ Model load error:", e)

@app.on_event("startup")
def _on_startup():
    warm_load()

# ---------- Small helpers ----------
def read_image_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return img

def downscale_max_side(img: np.ndarray, max_side=1024) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / float(s)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/model")
def model_info():
    return {"ok": True, "path": MODEL_PATH, "classes": CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = await file.read()
    img = read_image_to_bgr(data)
    img = downscale_max_side(img, max_side=1024)          # keep memory in check
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = MODEL.predict(
        source=img_rgb,
        imgsz=640,
        conf=0.4,
        iou=0.6,
        device="cpu",
        verbose=False,
        save=False,
        stream=False,
        half=False,     # keep False on CPU
        workers=0,
    )

    dets = []
    r0 = results[0]
    if r0.boxes is not None and len(r0.boxes) > 0:
        for b in r0.boxes.cpu().numpy():
            cls_id = int(b.cls[0])
            dets.append({
                "cls": cls_id,
                "name": CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else str(cls_id),
                "conf": float(b.conf[0]),
                "xyxy": [float(x) for x in b.xyxy[0]],
            })

    return {"ok": True, "detections": dets, "model": MODEL_PATH}

