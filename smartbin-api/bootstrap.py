import os, urllib.request, pathlib

def ensure_model():
    dst = pathlib.Path("/data/best.pt")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = os.environ.get("SMARTBIN_MODEL", "").strip()
    if not src:
        raise RuntimeError("SMARTBIN_MODEL env missing (local path or https URL)")
    if src.startswith("http"):
        if not dst.exists():
            print(f"Downloading model from {src} -> {dst}")
            urllib.request.urlretrieve(src, dst)
        return str(dst)
    return src
