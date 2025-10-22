from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, cv2, numpy as np
from ultralytics import YOLO
from bootstrap import ensure_model

app = FastAPI(title="SmartBin API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Load model from SMARTBIN_MODEL (local path or URL via bootstrap.ensure_model) ----
MODEL_PATH = os.environ.get("SMARTBIN_MODEL", "").strip() or None
try:
    MODEL_PATH = ensure_model()  # downloads if SMARTBIN_MODEL is a URL
    MODEL = YOLO(MODEL_PATH)
    LOADED_PATH = MODEL_PATH
except Exception as e:
    MODEL = None
    LOADED_PATH = None
    print("Model load error:", e)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/model")
def model_info():
    # report which path we tried and whether it’s loaded
    candidates = [os.environ.get("SMARTBIN_MODEL", "").strip()]
    return {"loaded_path": LOADED_PATH, "candidates": [c for c in candidates if c]}

# ---------- Robust image loader (drop alpha, clamp huge images) ----------
MAX_LONG_SIDE = 1600
def load_image_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    m = max(h, w)
    if m > MAX_LONG_SIDE:
        s = MAX_LONG_SIDE / float(m)
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

# ---------- Class thresholds & German bin mapping ----------
CLASS_THRESH = {
    "paper_cardboard": 0.25,
    "lvp_plastic_metal": 0.28,  # help cartons
    "glass_white": 0.20,        # help white-glass recall
    "glass_brown": 0.25,
    "glass_green": 0.25,
    "residual": 0.25,
    "battery": 0.25,
}

CLASS_TO_BIN = {
    "paper_cardboard": {
        "bin": "Papiertonne (Blue paper bin)",
        "note": "Clean, dry paper & cardboard. No food stains."
    },
    "lvp_plastic_metal": {
        "bin": "Gelber Sack / Gelbe Tonne (Lightweight packaging)",
        "note": "Packaging of plastic/metal/composites incl. drink cartons. Empty, not spotless."
    },
    "glass_white": {"bin": "Weißglas-Container", "note": "White/clear glass only. No lids/ceramics."},
    "glass_brown": {"bin": "Braunglas-Container", "note": "Brown glass bottles/jars."},
    "glass_green": {"bin": "Grünglas-Container", "note": "Green glass bottles/jars."},
    "residual": {"bin": "Restmüll (Residual waste)", "note": "Dirty/greasy paper, hygiene items, mixed trash."},
    "battery": {"bin": "Batteriesammelbox / Händler-Rücknahme", "note": "Hazardous: return to stores/collection points."},
}

# ---------- Predict (boxes) ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        return {"detections": [], "loaded_path": LOADED_PATH, "error": "model_not_loaded"}
    img = load_image_bytes(await file.read())

    # pass 1: normal; pass 2: bigger+augment if nothing found
    def run(conf, imgsz, augment=False):
        r = MODEL.predict(source=img, imgsz=imgsz, conf=conf, iou=0.6, augment=augment, verbose=False)[0]
        dets=[]
        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            names = MODEL.names
            for b in r.boxes:
                cls_id = int(b.cls.item()); conf_ = float(b.conf.item())
                x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                dets.append({"name": names[cls_id], "conf": conf_, "box": [x1,y1,x2,y2]})
        return dets

    dets = run(0.05, 640, False)
    if not dets:
        dets = run(0.001, 960, True)

    filtered = [d for d in dets if d["conf"] >= CLASS_THRESH.get(d["name"], 0.25)]
    if not filtered and dets:
        top = max(dets, key=lambda x: x["conf"])
        if top["conf"] >= 0.12:
            filtered = [top]

    return {"detections": filtered, "loaded_path": LOADED_PATH}

# ---------- Suggest (top-1 -> bin recommendation) ----------
@app.post("/suggest")
async def suggest(file: UploadFile = File(...)):
    # reuse predict logic
    resp = await predict(file)
    dets = resp.get("detections", []) if isinstance(resp, dict) else []
    if not dets:
        return {"suggestion": {"ok": False, "reason": "no_detections"}, "detections": [], "model": LOADED_PATH}
    top = max(dets, key=lambda d: d["conf"])
    info = CLASS_TO_BIN.get(top["name"], {"bin": "Unknown", "note": ""})
    return {
        "suggestion": {
            "ok": True,
            "top_class": top["name"],
            "confidence": round(float(top["conf"]), 3),
            "bin": info["bin"],
            "note": info["note"],
        },
        "detections": dets,
        "model": LOADED_PATH,
    }

