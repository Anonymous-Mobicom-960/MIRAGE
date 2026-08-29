"""
Tier 1 -> Tier 2 local link server (Pi/phone Phase 1, plan section 2.1).

Serves de-identified dense-export clip bundles (see pipeline.py's
--export-dir output) to the phone over local WiFi, via TOFU-pinned HTTPS
by default (see cert.py) -- pass --http to fall back to plain HTTP for
quick local testing. mDNS advertisement is a separate later pass (same
endpoint contract either way).

Directory layout expected under --root:
    <root>/<clip_id>/manifest.json
    <root>/<clip_id>/<any file the manifest lists>

Endpoints:
    GET  /clips                        -> list of {clip_id, status, ...}
    GET  /clips/{clip_id}/manifest     -> the clip's manifest.json, plus a
                                           "files" list with per-file sha256
    GET  /clips/{clip_id}/files/{name} -> the file's bytes, Range-aware
    POST /clips/{clip_id}/ack          -> phone confirms receipt; clip is
                                           marked acknowledged (not deleted
                                           here -- deletion policy is a
                                           later decision, kept explicit)

Usage:
    python -m tier1_link.server --root tier2_export_root --port 8443
    python -m tier1_link.server --root tier2_export_root --port 8000 --http
"""
import argparse
import hashlib
import json
import os
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from .cert import DEFAULT_CERT_PATH, DEFAULT_KEY_PATH, ensure_cert
from .discovery import Advertiser

app = FastAPI(title="MIRAGE Tier1 Link Server")

EXPORT_ROOT = None  # set in main() from --root
_acked_clips = set()


def _clip_dir(clip_id: str) -> str:
    path = os.path.join(EXPORT_ROOT, clip_id)
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail=f"clip '{clip_id}' not found")
    return path


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@app.get("/clips")
def list_clips():
    if not os.path.isdir(EXPORT_ROOT):
        return []
    out = []
    for clip_id in sorted(os.listdir(EXPORT_ROOT)):
        manifest_path = os.path.join(EXPORT_ROOT, clip_id, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        out.append({
            "clip_id": clip_id,
            "status": "acknowledged" if clip_id in _acked_clips else "ready",
            "total_frames": manifest.get("total_frames"),
            "num_slots": manifest.get("num_slots"),
        })
    return out


@app.get("/clips/{clip_id}/manifest")
def get_manifest(clip_id: str):
    d = _clip_dir(clip_id)
    manifest_path = os.path.join(d, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise HTTPException(status_code=404, detail="manifest.json missing for this clip")
    with open(manifest_path) as f:
        manifest = json.load(f)

    files = []
    for name in sorted(os.listdir(d)):
        if name == "manifest.json":
            continue
        full = os.path.join(d, name)
        if os.path.isfile(full):
            files.append({
                "name": name,
                "size": os.path.getsize(full),
                "sha256": _sha256_of(full),
            })
    manifest["files"] = files
    return manifest


@app.get("/clips/{clip_id}/files/{name}")
def get_file(clip_id: str, name: str, request: Request):
    d = _clip_dir(clip_id)
    # Reject any name that isn't a plain filename within the clip dir
    # (no path separators) -- stops a crafted name from escaping via ../.
    if name != os.path.basename(name):
        raise HTTPException(status_code=400, detail="invalid file name")
    path = os.path.join(d, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"file '{name}' not found in clip '{clip_id}'")
    # FileResponse handles Range requests natively (chunked/resumable pull).
    return FileResponse(path, filename=name)


@app.post("/clips/{clip_id}/ack")
def ack_clip(clip_id: str):
    _clip_dir(clip_id)  # 404s if the clip doesn't exist
    _acked_clips.add(clip_id)
    return JSONResponse({"clip_id": clip_id, "status": "acknowledged"})


def _ensure_device_id(path: str = "tier1_link_device_id.txt") -> str:
    if os.path.exists(path):
        return open(path).read().strip()
    device_id = str(uuid.uuid4())
    with open(path, "w") as f:
        f.write(device_id)
    return device_id


def main():
    global EXPORT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory containing <clip_id>/ subfolders")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--http", action="store_true",
                     help="Serve plain HTTP instead of TOFU-pinned HTTPS (dev/back-compat only).")
    ap.add_argument("--cert", default=DEFAULT_CERT_PATH)
    ap.add_argument("--key", default=DEFAULT_KEY_PATH)
    ap.add_argument("--no-mdns", action="store_true", help="Disable mDNS advertisement.")
    ap.add_argument("--mdns-name", default="mirage-tier1",
                     help="mDNS instance name, e.g. mirage-tier1.local.")
    args = ap.parse_args()

    EXPORT_ROOT = os.path.abspath(args.root)
    print(f"Serving clip exports from: {EXPORT_ROOT}")

    device_id = _ensure_device_id()

    advertiser = None
    if not args.no_mdns:
        advertiser = Advertiser(args.mdns_name, args.port, device_id, uses_https=not args.http)
        advertiser.start()

    try:
        if args.http:
            print("  WARNING: --http requested, serving PLAIN HTTP (no transport security).")
            uvicorn.run(app, host=args.host, port=args.port)
        else:
            fingerprint = ensure_cert(args.cert, args.key)
            print("=" * 60)
            print("  TOFU pairing fingerprint (SHA-256):")
            print(f"  {fingerprint}")
            print("  Enter this exact value in the phone app on first connect.")
            print("=" * 60)
            uvicorn.run(app, host=args.host, port=args.port,
                         ssl_certfile=args.cert, ssl_keyfile=args.key)
    finally:
        if advertiser is not None:
            advertiser.stop()


if __name__ == "__main__":
    main()
