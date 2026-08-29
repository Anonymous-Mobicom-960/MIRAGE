import os
import cv2
import numpy as np
import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization


def generate_ttp_keypair():
    # RSA-4096, matching the original MIRAGE paper (Section IV.A). ONLY the
    # real Tier 3 TTP server (src/tier3_ttp/server.py) should ever call this
    # -- it's how the TTP mints its own keypair at startup. Tier 1
    # (pipeline.py) must NEVER call this: generating a keypair locally means
    # holding the private key right next to the data it's supposed to
    # protect, which defeats the entire point of a third-party consent
    # server. Tier 1 calls fetch_ttp_public_key() instead.
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    return private_key, private_key.public_key()


def fetch_ttp_public_key(server_url: str, verify_tls: bool = True):
    """Fetches the real Tier 3 TTP's RSA public key over HTTP(S) -- the only
    way Tier 1 should ever obtain a key to wrap AES keys with. Raises on any
    network/HTTP failure (no silent local-keypair fallback -- if the real
    TTP isn't reachable, that's a hard error, not a reason to simulate one).
    verify_tls=False matches the TOFU-pinned server's self-signed cert not
    being CA-validated (see tier1_link/cert.py); real trust here should
    ultimately come from TOFU fingerprint pinning, not skipped entirely --
    same caveat as the rest of this codebase's current TLS handling."""
    resp = requests.get(f"{server_url}/v1/public-key", verify=verify_tls)
    resp.raise_for_status()
    pem = resp.json()["public_key_pem"].encode("ascii")
    return serialization.load_pem_public_key(pem)


def rsa_encrypt_key(aes_key: bytes, rsa_public_key) -> bytes:
    return rsa_public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        )
    )


def rsa_decrypt_key(encrypted_aes_key: bytes, rsa_private_key) -> bytes:
    # TTP-side counterpart to rsa_encrypt_key -- only the TTP ever holds the
    # private key, so this is the one function in this module that never
    # runs on the wearer's device.
    return rsa_private_key.decrypt(
        encrypted_aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        )
    )


def aes_gcm_encrypt(aes_key: bytes, plaintext: bytes):
    nonce = os.urandom(12)
    return nonce, AESGCM(aes_key).encrypt(nonce, plaintext, None)


def aes_gcm_decrypt(aes_key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    return AESGCM(aes_key).decrypt(nonce, ciphertext, None)


def encode_crop(crop_bgr: np.ndarray, quality: int = 70) -> bytes:
    # Q=70 is the deployed operating point: the knee of the rate-distortion sweep
    # over the crop archive (SSIM 0.9793 / PSNR 38.31 dB at 60.5% less payload than Q=95).
    success, buf = cv2.imencode('.jpg', crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()
