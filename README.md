# SmartBin ♻️

YOLOv8-based waste classifier aligned with German recycling rules 🇩🇪.

[![CI](https://github.com/ghitaik/smartbin/actions/workflows/ci.yml/badge.svg)](https://github.com/ghitaik/smartbin/actions)


**Live API:** https://smartbin-api-4ycp.onrender.com  
**Web Demo:** https://smartbin-virid.vercel.app

---

## 🧠 Overview

SmartBin automatically detects waste items and tells you which **German recycling bin** they belong to.

It uses a lightweight YOLOv8 model trained on common household waste categories and runs fully on CPU — optimized for Render’s free tier.

---

## 📂 Repo structure

smartbin/
├─ smartbin-api/ # FastAPI backend
│ ├─ app.py # /health, /model, /predict
│ ├─ bootstrap.py # Model fetch (writes to /tmp for Render)
│ └─ requirements.txt # CPU wheels (small deploys)
├─ smartbin-web/ # Static frontend (Vercel)
│ └─ index.html
├─ examples/ # Test images
├─ tests/ # Light API smoke tests
├─ render.yaml
└─ README.md

---

## 🚀 Local API quickstart

```bash
cd smartbin-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export SMARTBIN_MODEL="https://<public-url>/best.pt"  # or absolute local path
uvicorn app:app --reload --port 8000

Test locally

  -GET http://127.0.0.1:8000/health → { "ok": true }

  -GET http://127.0.0.1:8000/model → model info

  -POST http://127.0.0.1:8000/predict → upload file

  ---

🧪 Web demo (Vercel)

Open smartbin-web/index.html
 in your browser
or visit the live demo:
👉 https://smartbin-virid.vercel.app/

Set your API URL to
https://smartbin-api-4ycp.onrender.com
then upload an image — e.g., a battery, bottle, or cardboard box.

---

♻️ German bin mapping

| Class               | Bin                               |
| ------------------- | --------------------------------- |
| `paper_cardboard`   | 🟦 Blue bin (Papier)              |
| `lvp_plastic_metal` | 🟨 Yellow bin (Gelber Sack/Tonne) |
| `glass_white`       | ⚪ White glass container           |
| `glass_brown`       | 🟤 Brown glass container          |
| `glass_green`       | 🟢 Green glass container          |
| `residual`          | ⚫ Restmüll                        |
| `battery`           | 🔋 Sondermüll (Hazardous waste)   |

---

🧰 CI

GitHub Actions run a lightweight smoke test with SMARTBIN_SKIP_LOAD=1
so YOLO/Torch aren’t downloaded during CI builds.

---

📜 License

MIT — see LICENSE

https://smartbin-virid.vercel.app/
