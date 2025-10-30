# 🧪 SmartBin Example Images

This folder contains small **demo images** you can use to test the SmartBin API (`/predict` endpoint).

| File name | Expected class | Notes |
|------------|----------------|-------|
| `battery.png` | `battery` | Should trigger battery detection |
| `paper_cardboard.jpg` | `paper_cardboard` | Paper or carton material |
| `glass_brown.png` | `glass_brown` | Brown glass bottle |
| `glass_green.png` | `glass_green` | Green glass bottle |
| `glass_white.png` | `glass_white` | Transparent glass bottle |
| `plastic_bottle.png` | `lvp_plastic_metal` | Typical PET bottle |
| `residual_bread.png` | `residual` | Organic waste |
| `residual_rest.png` | `residual` | Mixed waste |
| `carton.png` | `paper_cardboard` | Cardboard box |

---

### 🔍 Test directly from your terminal

```bash
curl -s -X POST \
  -H "Accept: application/json" \
  -F "file=@examples/battery.png;type=image/png" \
  https://smartbin-api-4ycp.onrender.com/predict | jq .

