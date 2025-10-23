import os, urllib.request, pathlib

# Allow overriding destination; default to /tmp which is writable on Render/most hosts
DST_DEFAULT = os.environ.get("SMARTBIN_DST", "/tmp/best.pt")

def ensure_model():
    """
    Returns a local path to the model.
    - If SMARTBIN_MODEL is an HTTP(S) URL, download once to a writable location (default /tmp/best.pt).
    - If SMARTBIN_MODEL is a local path, just return it.
    """
    src = os.environ.get("SMARTBIN_MODEL", "").strip()
    if not src:
        raise RuntimeError("SMARTBIN_MODEL env missing (local path or https URL)")

    dst = pathlib.Path(DST_DEFAULT)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.startswith("http"):
        if not dst.exists():
            print(f"Downloading model from {src} -> {dst}")
            urllib.request.urlretrieve(src, dst)
        return str(dst)

    # local path case
    return src

