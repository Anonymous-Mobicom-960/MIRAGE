"""
mDNS (DNS-SD) advertisement for the Tier1 link server (plan section 2.1).
Pure convenience -- lets the phone find "the MIRAGE device on this
network" instead of the user typing an IP. Not a security boundary:
TOFU-pinned HTTPS (see cert.py) is what actually protects the connection;
mDNS just answers "where is it," the same way Avahi would on the real Pi.

Service type: _mirage._tcp.local. -- a private, unregistered service
type, fine for a research rig (real products register with IANA).
"""
import socket

from zeroconf import ServiceInfo, Zeroconf

SERVICE_TYPE = "_mirage._tcp.local."


def _local_ip() -> str:
    """Best-effort local LAN IP (the address a phone on the same network
    would actually use to reach this machine), without needing real
    outbound traffic -- connecting a UDP socket doesn't send packets."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Advertiser:
    def __init__(self, instance_name: str, port: int, device_id: str, uses_https: bool):
        self._zc = Zeroconf()
        ip = _local_ip()
        self._info = ServiceInfo(
            SERVICE_TYPE,
            f"{instance_name}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=port,
            properties={
                "device_id": device_id,
                "scheme": "https" if uses_https else "http",
            },
            server=f"{instance_name}.local.",
        )
        self._ip = ip

    def start(self):
        self._zc.register_service(self._info)
        print(f"  mDNS: advertising as {self._info.name} @ {self._ip}:{self._info.port}")

    def stop(self):
        self._zc.unregister_service(self._info)
        self._zc.close()
