# app.py — SmartBin API (health + model info + predict + simple UI)
import io
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import cv2

CLASSES = [
    "paper_cardboard","lvp_plastic_metal","glass_white",
    "glass_brown","glass_green","residual","battery",
]

MODEL_CANDIDATES = [
    "../smartbin-train/latest_best.pt",
    "../smartbin-train/runs/detect/mac_ft_boosters/weights/best.pt",
    "../smartbin-train/best.pt",
]

GLOBAL_CONF = 0.30
PER_CLASS_MIN = {
    "battery": 0.60,
    "residual": 0.80,
    "lvp_plastic_metal": 0.30,
    "paper_cardboard": 0.30,
}

DEBUG_HEUR = True  # include debug fields in JSON

# ---------- Helpers ----------
def _crop(arr_rgb: np.ndarray, box: Dict[str,float], pad: float = 0.0) -> np.ndarray:
    h, w = arr_rgb.shape[:2]
    x1 = max(int((box["x1"] - pad) * w), 0)
    y1 = max(int((box["y1"] - pad) * h), 0)
    x2 = min(int((box["x2"] + pad) * w), w-1)
    y2 = min(int((box["y2"] + pad) * h), h-1)
    if x2 <= x1 or y2 <= y1: return np.zeros((0,0,3), dtype=np.uint8)
    return arr_rgb[y1:y2, x1:x2][:,:,::-1].copy()  # RGB->BGR

def glass_color_from_hsv(bgr_crop: np.ndarray) -> str:
    if bgr_crop.size == 0: return "glass_white"
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    H = hsv[...,0].astype(np.float32) * (360.0/180.0)
    S = hsv[...,1].astype(np.float32)/255.0
    V = hsv[...,2].astype(np.float32)/255.0
    mask = (S > 0.25) & (V > 0.20)
    if mask.sum() < 50: return "glass_white"
    h_med = float(np.median(H[mask]))
    if 70.0 <= h_med <= 170.0: return "glass_green"
    if 10.0 <= h_med <= 50.0:  return "glass_brown"
    return "glass_white"

def edge_density_gray(bgr_crop: np.ndarray) -> float:
    if bgr_crop.size == 0: return 1.0
    g = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    e = cv2.Canny(g, 60, 120)
    return float((e > 0).mean())

# ---- Paper sheet heuristic (strict) ----
def paper_likelihood(bgr_crop: np.ndarray) -> dict:
    if bgr_crop.size == 0:
        return {"is_paper": False, "white_ratio":0, "colorful_ratio":1, "tex":1, "meanS":1, "meanV":0}
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    S = hsv[...,1].astype(np.float32)/255.0
    V = hsv[...,2].astype(np.float32)/255.0
    whiteish = (S < 0.22) & (V > 0.80)
    colorful = (S > 0.40) & (V > 0.25)
    white_ratio    = float(whiteish.mean()) if whiteish.size else 0.0
    colorful_ratio = float(colorful.mean()) if colorful.size else 1.0
    tex            = edge_density_gray(bgr_crop)
    meanS, meanV   = float(S.mean()), float(V.mean())
    is_paper = (white_ratio > 0.70) and (colorful_ratio < 0.12) and (tex < 0.10) and (meanS < 0.22) and (meanV > 0.80)
    return {"is_paper": bool(is_paper), "white_ratio": white_ratio,
            "colorful_ratio": colorful_ratio, "tex": tex, "meanS": meanS, "meanV": meanV}

# ---- Cardboard heuristic (v2: looser + HSV fallback) ----
def cardboard_likelihood(bgr_crop: np.ndarray) -> dict:
    """
    Identify brown cardboard (pizza/shoe boxes, corrugated look).
    """
    if bgr_crop.size == 0:
        return {"is_cardboard": False, "brown_ratio":0, "white_ratio":0, "tex":1, "grease_ratio":0, "meanS":1, "meanV":0}
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    H = hsv[...,0].astype(np.float32) * (360.0/180.0)
    S = hsv[...,1].astype(np.float32)/255.0
    V = hsv[...,2].astype(np.float32)/255.0

    brownish = (H >= 10.0) & (H <= 55.0) & (S > 0.18) & (V > 0.30) & (V < 0.90)
    whiteish = (S < 0.18) & (V > 0.85)
    brown_ratio = float(brownish.mean()) if brownish.size else 0.0
    white_ratio = float(whiteish.mean()) if whiteish.size else 0.0

    tex = edge_density_gray(bgr_crop)
    grease = (V < 0.28)
    grease_ratio = float(grease.mean()) if grease.size else 0.0
    meanS, meanV = float(S.mean()), float(V.mean())

    is_cardboard = (
        (brown_ratio - white_ratio > 0.12) and
        (0.33 <= meanV <= 0.88) and
        (meanS >= 0.16) and
        (0.035 <= tex <= 0.40)
    )
    return {"is_cardboard": bool(is_cardboard), "brown_ratio": brown_ratio,
            "white_ratio": white_ratio, "tex": tex, "grease_ratio": grease_ratio,
            "meanS": meanS, "meanV": meanV}

def cardboard_hsv_vote(bgr_crop: np.ndarray) -> dict:
    """
    Simple HSV vote: median hue/sat in brown range, not too white.
    This is a fallback if the main cardboard_likelihood is borderline.
    """
    if bgr_crop.size == 0:
        return {"vote": False, "h_med":0, "s_med":0, "v_med":0, "white_ratio":0}
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    H = hsv[...,0].astype(np.float32) * (360.0/180.0)
    S = hsv[...,1].astype(np.float32)/255.0
    V = hsv[...,2].astype(np.float32)/255.0
    h_med = float(np.median(H))
    s_med = float(np.median(S))
    v_med = float(np.median(V))
    whiteish = (S < 0.18) & (V > 0.85)
    white_ratio = float(whiteish.mean()) if whiteish.size else 0.0

    vote = (15.0 <= h_med <= 50.0) and (s_med >= 0.20) and (white_ratio < 0.45) and (0.30 <= v_med <= 0.90)
    return {"vote": bool(vote), "h_med": h_med, "s_med": s_med, "v_med": v_med, "white_ratio": white_ratio}

# ---------- FastAPI ----------
app = FastAPI(title="SmartBin API", version="0.3.7")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
LOADED_PATH = None

def get_model():
    global _model, LOADED_PATH
    if _model is None:
        from ultralytics import YOLO
        last_err = None
        for p in MODEL_CANDIDATES:
            try:
                _model = YOLO(p); LOADED_PATH = p
                print(f"[SmartBin] Loaded model: {p}"); break
            except Exception as e:
                last_err = e; continue
        if _model is None:
            raise RuntimeError(f"No model weights found in {MODEL_CANDIDATES}. Last error: {last_err}")
    return _model

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "classes": CLASSES}

@app.get("/model")
def model_info() -> Dict[str, Any]:
    return {"loaded_path": LOADED_PATH, "candidates": MODEL_CANDIDATES}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    qp = dict(request.query_params)
    debug = DEBUG_HEUR if "debug" not in qp else (qp.get("debug","1") != "0")

    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]

    model = get_model()
    res = model.predict(source=arr, conf=GLOBAL_CONF, iou=0.5, verbose=False)[0]

    dets: List[Dict[str, Any]] = []
    if res.boxes is not None:
        for b in res.boxes:
            conf = float(b.conf[0]); cls_id = int(b.cls[0])
            name = model.names.get(cls_id, str(cls_id))
            if conf < PER_CLASS_MIN.get(name, GLOBAL_CONF): continue

            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            det = {"class": name, "confidence": round(conf, 4),
                   "bbox": {"x1": x1/W, "y1": y1/H, "x2": x2/W, "y2": y2/H}}

            # 1) Glass color correction
            if det["class"].startswith("glass_"):
                crop_bgr = _crop(arr, det["bbox"], pad=0.01)
                det["class"] = glass_color_from_hsv(crop_bgr)

            # 2) Cardboard vs carton (pizza box etc.)
            if det["class"] == "lvp_plastic_metal" and det["confidence"] < 0.95:
                bx = det["bbox"]; area = (bx["x2"]-bx["x1"]) * (bx["y2"]-bx["y1"])
                crop_bgr = _crop(arr, bx)
                cb = cardboard_likelihood(crop_bgr)
                hsvv = cardboard_hsv_vote(crop_bgr)

                if area > 0.12 and (cb["is_cardboard"] or hsvv["vote"]):
                    # greasy cardboard -> residual, else paper_cardboard
                    if cb["grease_ratio"] > 0.12:
                        det["class"] = "residual"
                    else:
                        det["class"] = "paper_cardboard"
                    det["confidence"] = max(float(det["confidence"]), 0.60)
                    if debug: 
                        det["cardboard_debug"] = {"area": round(area,3), **{k:round(v,3) if isinstance(v,(int,float)) else v for k,v in cb.items()}}
                        det["cardboard_debug_hsv"] = {k: (round(v,3) if isinstance(v,(int,float)) else v) for k,v in hsvv.items()}

            # 3) Paper sheet vs residual (strict)
            if det["class"] == "residual" and det["confidence"] < 0.96:
                bx = det["bbox"]; area = (bx["x2"]-bx["x1"]) * (bx["y2"]-bx["y1"])
                crop_bgr = _crop(arr, bx)
                pl = paper_likelihood(crop_bgr)
                if (area > 0.70) and pl["is_paper"]:
                    det["class"] = "paper_cardboard"
                    det["confidence"] = max(float(det["confidence"]), 0.60)
                    if debug: det["paper_debug"] = {"area": round(area,3), **{k:round(v,3) for k,v in pl.items()}}
                else:
                    if debug: det["paper_debug"] = {"area": round(area,3), **{k:round(v,3) for k,v in pl.items()}}

            dets.append(det)

    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return {"detections": dets, "model": LOADED_PATH}

# ---- Minimal UI ----
@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!doctype html><html><head><meta charset="utf-8"/><title>SmartBin Demo</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;max-width:900px;margin:24px auto;padding:0 16px}
.panel{border:1px solid #ddd;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 2px 8px rgba(0,0,0,.04)}
#stage{position:relative;display:inline-block;max-width:100%}#img{max-width:100%;display:block}#canvas{position:absolute;left:0;top:0}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}button{padding:8px 14px;border-radius:10px;border:1px solid #ccc;background:#f6f6f6;cursor:pointer}
#log{white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:12px}.legend{display:flex;gap:10px;flex-wrap:wrap;margin-top:6px}.tag{padding:2px 6px;border-radius:8px;border:1px solid #ddd;font-size:12px}
</style></head><body>
<h1>SmartBin – Image Test</h1>
<div class="panel"><div class="row"><input id="file" type="file" accept="image/*"/><button id="run">Predict</button><span id="status"></span></div>
<div id="stage"><img id="img"/><canvas id="canvas"></canvas></div><div class="legend" id="legend"></div></div>
<div class="panel"><h3>Model</h3><div id="model">–</div></div>
<div class="panel"><h3>Raw response</h3><div id="log">–</div></div>
<script>
const CLRS={paper_cardboard:'#2b8a3e',lvp_plastic_metal:'#1971c2',glass_white:'#868e96',glass_brown:'#a0613b',glass_green:'#2f9e44',residual:'#e03131',battery:'#ae3ec9'};
const img=document.getElementById('img'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),fileInput=document.getElementById('file'),runBtn=document.getElementById('run'),statusEl=document.getElementById('status'),logEl=document.getElementById('log'),modelEl=document.getElementById('model');
function setLegend(){document.querySelector('.legend').innerHTML=Object.entries(CLRS).map(([k,v])=>`<span class="tag" style="border-color:${v};color:${v}">${k}</span>`).join(' ');} setLegend();
async function refreshModel(){try{const r=await fetch('/model');const j=await r.json();modelEl.textContent=j.loaded_path||'(not yet loaded)';}catch(e){modelEl.textContent='(error)';}} refreshModel();
fileInput.addEventListener('change',()=>{const f=fileInput.files[0];if(!f)return;const url=URL.createObjectURL(f);img.onload=()=>{fit();draw([])};img.src=url;});
runBtn.addEventListener('click',async()=>{const f=fileInput.files[0];if(!f){alert('Choose an image first');return;}statusEl.textContent='Predicting...';const fd=new FormData();fd.append('file',f);const r=await fetch('/predict?debug=1',{method:'POST',body:fd});const j=await r.json();logEl.textContent=JSON.stringify(j,null,2);fit();draw(j.detections||[]);statusEl.textContent='Done';refreshModel();});
function fit(){canvas.width=img.clientWidth;canvas.height=img.clientHeight;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.lineWidth=2;ctx.font='14px ui-monospace, monospace';}
function draw(dets){ctx.clearRect(0,0,canvas.width,canvas.height);for(const d of dets){const c=CLRS[d.class]||'#222';const x1=d.bbox.x1*canvas.width,y1=d.bbox.y1*canvas.height,x2=d.bbox.x2*canvas.width,y2=d.bbox.y2*canvas.height;const w=x2-x1,h=y2-y1;ctx.strokeStyle=c;ctx.strokeRect(x1,y1,w,h);const label=`${d.class} ${Math.round(d.confidence*100)/100}`;const pad=4,tw=ctx.measureText(label).width+pad*2,th=18;ctx.fillStyle=c;ctx.fillRect(x1,y1-th,tw,th);ctx.fillStyle='white';ctx.fillText(label,x1+pad,y1-4);}}
</script></body></html>
"""
