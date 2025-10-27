from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os, io, time, pathlib, urllib.request
import numpy as np
import cv2
from ultralytics import YOLO

# -----------------------------
# CORS (allow web app to call API)
# -----------------------------
app = FastAPI(title="SmartBin API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later to your Vercel URL if you want
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Global model cache
# -----------------------------
MODEL = None            # YOLO() object
LOADED_PATH = None      # Resolved file actually loaded (path on disk)

def _resolve_model_path() -> str:
    """
    Reads SMARTBIN_MODEL env. If it's an HTTPS URL, download to /tmp/smartbin/best.pt.
    Otherwise, treat it as a local absolute path.
    """
    src = os.environ.get("SMARTBIN_MODEL", "").strip()
    if not src:
        raise RuntimeError("SMARTBIN_MODEL env missing (local path or https URL)")

    if src.lower().startswith(("http://", "https://")):
        dst = pathlib.Path("/tmp/smartbin/best.pt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            print(f"Downloading model from {src} -> {dst}")
            urllib.request.urlretrieve(src, dst)
        return str(dst)

    # local path
    if not os.path.exists(src):
        raise FileNotFoundError(f"SMARTBIN_MODEL path not found: {src}")
    return src

def get_model():
    """
    Loads the YOLO model once and reuses it (warm cache).
    """
    global MODEL, LOADED_PATH
    resolved = _resolve_model_path()
    if MODEL is None or LOADED_PATH != resolved:
        t0 = time.time()
        MODEL = YOLO(resolved)
        LOADED_PATH = resolved
        print(f"✅ YOLO model loaded from {resolved} in {time.time()-t0:.2f}s")
    return MODEL

# -----------------------------
# Startup: warm-load model
# -----------------------------
@app.on_event("startup")
def _warmup():
    try:
        m = get_model()        # force load at boot
        _ = m.names            # touch attribute to keep ready
        print("🔹 Model warm-loaded at startup.")
    except Exception as e:
        # Non-fatal: we still lazy-load on first real request
        print(f"⚠️ Warm-load failed (will lazy-load later): {e}")

# -----------------------------
# Schemas
# -----------------------------
class PredictOut(BaseModel):
    detections: List[Dict[str, Any]]
    model: str

class SuggestOut(BaseModel):
    bin: str
    reason: str
    top_class: str
    detections: List[Dict[str, Any]]

# -----------------------------
# Util: read UploadFile -> OpenCV image (BGR)
# -----------------------------
def _file_to_bgr(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/model")
def model_info():
    m = get_model()
    # m.names is a dict like {0:'paper_cardboard', ...} -> convert to list by index order
    idx_to_name = [m.names[i] for i in sorted(m.names.keys())]
    return {"ok": True, "path": LOADED_PATH, "classes": idx_to_name}

@app.post("/predict", response_model=PredictOut)
def predict(file: UploadFile = File(...)):
    img = _file_to_bgr(file)
    m = get_model()

    # CPU inference on Render free tier
    res = m.predict(
        source=img,
        imgsz=640,
        conf=0.25,
        iou=0.6,
        max_det=10,
        device="cpu",
        verbose=False
    )[0]

    dets: List[Dict[str, Any]] = []
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        conf = float(b.conf[0].item())
        cls_id = int(b.cls[0].item())
        cls_name = m.names.get(cls_id, str(cls_id))
        dets.append({
            "cls_id": cls_id,
            "cls_name": cls_name,
            "conf": round(conf, 3),
            "xyxy": [x1, y1, x2, y2],
        })

    return {"detections": dets, "model": LOADED_PATH}

@app.post("/suggest", response_model=SuggestOut)
def suggest(file: UploadFile = File(...)):
    img = _file_to_bgr(file)
    m = get_model()

    res = m.predict(
        source=img,
        imgsz=640,
        conf=0.25,
        iou=0.6,
        max_det=10,
        device="cpu",
        verbose=False
    )[0]

    dets: List[Dict[str, Any]] = []
    best = None
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        conf = float(b.conf[0].item())
        cls_id = int(b.cls[0].item())
        cls_name = m.names.get(cls_id, str(cls_id))
        item = {
            "cls_id": cls_id,
            "cls_name": cls_name,
            "conf": round(conf, 3),
            "xyxy": [x1, y1, x2, y2],
        }
        dets.append(item)
        if best is None or conf > best["conf"]:
            best = item

    # Default if nothing found
    if not best:
        return {
            "bin": "unknown",
            "reason": "No confident detections.",
            "top_class": "none",
            "detections": dets
        }

    name = best["cls_name"]
    # Simple German-rule mapping
    if name in ("paper_cardboard", "paper", "cardboard"):
        bin_name = "Papier (Blue bin)"
        why = "Paper and clean cardboard go to the blue bin."
    elif name == "lvp_plastic_metal":
        bin_name = "Gelber Sack/Tonne (Yellow bin)"
        why = "Light packaging (plastic/metal/TetraPak) goes to the yellow bin."
    elif name.startswith("glass_"):
        color = name.split("_", 1)[1]
        color_map = {"white": "white (clear)", "brown": "brown", "green": "green"}
        bin_name = f"Glass container ({color_map.get(color, color)})"
        why = "Glass is collected in color-sorted bottle banks."
    elif name == "battery":
        bin_name = "Battery collection point"
        why = "Batteries are hazardous—return to stores or collection points."
    else:
        bin_name = "Restmüll (Residual)"
        why = "If not recyclable, it goes to residual waste."

    return {
        "bin": bin_name,
        "reason": why,
        "top_class": name,
        "detections": dets
    }
