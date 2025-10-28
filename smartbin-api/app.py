from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os, gc, numpy as np, cv2

# model locator (downloads to /tmp on Render if SMARTBIN_MODEL is a URL)
try:
    from bootstrap import ensure_model
except Exception:
    def ensure_model():
        p = os.environ.get("SMARTBIN_MODEL", "")
        if not p:
            raise RuntimeError("SMARTBIN_MODEL env missing (local path or https URL)")
        return p

app = FastAPI(title="SmartBin API", version="1.0")

# Open CORS so Vercel web app can call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

MODEL = None
MODEL_PATH: Optional[str] = None
CLASS_NAMES = [
    "paper_cardboard",
    "lvp_plastic_metal",
    "glass_white",
    "glass_brown",
    "glass_green",
    "residual",
    "battery",
]

@app.on_event("startup")
def startup():
    """Warm-load YOLO on CPU for Render free tier."""
    global MODEL, MODEL_PATH
    if os.getenv("SMARTBIN_SKIP_LOAD") == "1":
        print("⚠️  SMARTBIN_SKIP_LOAD=1 -> skipping model load", flush=True)
        return
    MODEL_PATH = ensure_model()
    from ultralytics import YOLO  # import here to keep import light until needed
    MODEL = YOLO(MODEL_PATH)
    # CPU-friendly defaults
    MODEL.overrides.update(dict(imgsz=512, half=False, device="cpu", workers=0))
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
    model: Optional[str] = None
    error: Optional[str] = None

def _decode_image(b: bytes) -> np.ndarray:
    """Decode bytes → BGR ndarray, with safety limits."""
    MAX_BYTES = 8 * 1024 * 1024  # 8 MB cap to protect 512MB instance
    if len(b) > MAX_BYTES:
        raise ValueError(f"image too large: {len(b)} > {MAX_BYTES} bytes")

    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    # Aggressive downscale to save RAM
    h, w = img.shape[:2]
    s = max(h, w)
    if s > 1024:
        k = 1024 / s
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return img

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        return {"ok": False, "detections": [], "model": MODEL_PATH, "error": "model not loaded"}

    # Read bytes then close ASAP to free memory
    b = await file.read()
    await file.close()

    try:
        img = _decode_image(b)
    except Exception as e:
        return {"ok": False, "detections": [], "model": MODEL_PATH, "error": str(e)}

    try:
        results = MODEL.predict(
            source=img, imgsz=512, conf=0.25, iou=0.45, device="cpu", verbose=False
        )
        dets = []
        for r in results:
            bs = getattr(r, "boxes", None)
            if bs is None:
                continue
            for bx in bs:
                cls_id = int(bx.cls)
                dets.append(
                    {
                        "class": CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id),
                        "conf": float(bx.conf),
                        "xyxy": [float(x) for x in bx.xyxy[0].tolist()],
                    }
                )
        return {"ok": True, "detections": dets, "model": MODEL_PATH}
    except Exception as e:
        return {"ok": False, "detections": [], "model": MODEL_PATH, "error": str(e)}
    finally:
        try:
            del img, b, results
        except Exception:
            pass
        gc.collect()

