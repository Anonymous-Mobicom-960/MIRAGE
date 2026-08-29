"""
Self-signed TLS cert for the Tier1 link server's TOFU (Trust On First Use)
pairing model (plan section 2.1). There's no real certificate authority for
a benchtop Pi, so instead: the server generates one ECDSA P-256 keypair +
self-signed cert on first run and reuses it forever after; the phone pins
the cert's SHA-256 fingerprint the first time it connects and refuses to
talk to a server presenting a different fingerprint later. This stops an
impostor device from silently replacing the real Pi on the same network
after pairing -- it does NOT provide any protection on the very first
connection (trust is placed then, by definition of TOFU), and is a
deliberate simplification of the plan's "manufacturer-signed root key"
language, which has no real analog on a one-off research rig.

This key is separate from encryption.py's RSA-4096 TTP-wrapping key --
different purpose (transport-layer server identity vs. per-stream AES key
wrapping for the TTP), different algorithm, different lifetime.
"""
import datetime
import hashlib
import os

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

DEFAULT_CERT_PATH = "tier1_link_cert.pem"
DEFAULT_KEY_PATH = "tier1_link_key.pem"


def generate_self_signed_cert(cert_path: str, key_path: str, common_name: str = "mirage-tier1"):
    private_key = ec.generate_private_key(ec.SECP256R1())

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        # 10 years -- this is a benchtop device identity key, not meant to
        # rotate on a schedule; TOFU pinning means clients would need to
        # re-pair on rotation anyway, so long-lived avoids that friction.
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    with open(key_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    return cert


def cert_fingerprint_sha256(cert_path: str) -> str:
    with open(cert_path, "rb") as f:
        cert = x509.load_pem_x509_certificate(f.read())
    der = cert.public_bytes(serialization.Encoding.DER)
    return hashlib.sha256(der).hexdigest()


def ensure_cert(cert_path: str = DEFAULT_CERT_PATH, key_path: str = DEFAULT_KEY_PATH) -> str:
    """Generates a cert/key pair if missing, returns the cert's sha256 fingerprint."""
    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        generate_self_signed_cert(cert_path, key_path)
    return cert_fingerprint_sha256(cert_path)
