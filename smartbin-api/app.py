from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os, gc, numpy as np, cv2
from ultralytics import YOLO
from bootstrap import ensure_model

app = FastAPI(title="SmartBin API", version="1.0")

# open CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
MODEL_PATH = None
CLASS_NAMES = ["paper_cardboard","lvp_plastic_metal","glass_white","glass_brown","glass_green","residual","battery"]

@app.on_event("startup")
def warm_load():
    global MODEL, MODEL_PATH
    MODEL_PATH = ensure_model()
    MODEL = YOLO(MODEL_PATH)
    # CPU-friendly defaults for Render free tier
    MODEL.overrides.update(dict(imgsz=640, half=False, device="cpu", workers=0))
    print(f"✅ YOLO warm-loaded from {MODEL_PATH}", flush=True)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/model")
def model_info():
    return {"ok": True, "path": MODEL_PATH, "classes": CLASS_NAMES}

class PredictResponse(BaseModel):
    ok: bool
    detections: list
    model: str

def _decode_image(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    # downscale huge images to keep memory tiny
    h, w = img.shape[:2]
    s = max(h, w)
    if s > 1600:
        k = 1600 / s
        img = cv2.resize(img, (int(w*k), int(h*k)), interpolation=cv2.INTER_AREA)
    return img

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    b = await file.read()
    try:
        img = _decode_image(b)
    except Exception:
        return {"ok": False, "detections": [], "model": MODEL_PATH}

    results = MODEL.predict(source=img, imgsz=640, conf=0.25, iou=0.45, device="cpu", verbose=False)

    dets = []
    for r in results:
        for bx in r.boxes:
            cls_id = int(bx.cls)
            dets.append({
                "class": CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id),
                "conf": float(bx.conf),
                "xyxy": [float(x) for x in bx.xyxy[0].tolist()],
            })

    del results, img, b
    gc.collect()
    return {"ok": True, "detections": dets, "model": MODEL_PATH}
