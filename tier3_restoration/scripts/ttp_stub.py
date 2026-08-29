#!/usr/bin/env python3
"""LOCAL TTP STUB -- serves only GET /v1/public-key, the one endpoint Tier-1 calls
(`mirage/encryption.py:fetch_ttp_public_key`). The real Tier-3 consent server
(`src/tier3_ttp/server.py`) does not exist in this repo (shipped_pipeline_spec.md section 4);
this stands in for it so the AES-128-GCM per-person packets get a real RSA-4096 wrap and can
be DECRYPTED afterwards to demonstrate the consent tier.

TEST KEY ONLY -- the private half sits next to it on disk, so nothing here is a security claim.
    python ttp_stub.py <private_key.pem> [port]
"""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from cryptography.hazmat.primitives import serialization

if len(sys.argv) < 2:
    sys.exit("usage: ttp_stub.py <private_key.pem> [port]   (see tier3_restoration/README.md)")

KEY = serialization.load_pem_private_key(open(sys.argv[1], "rb").read(), password=None)
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8843
PEM = KEY.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii")


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") != "/v1/public-key":
            self.send_error(404); return
        b = json.dumps({"public_key_pem": PEM}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


print(f"TTP stub on :{PORT}  RSA-{KEY.key_size}", flush=True)
HTTPServer(("127.0.0.1", PORT), H).serve_forever()
