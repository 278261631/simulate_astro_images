#!/usr/bin/env python3
"""Self-test for the NINA Alpaca camera simulator (no live NINA required).

Exercises the full Alpaca HTTP pipeline with a stub NINA data source:
connect -> set window -> exposurestart -> poll state/imageready -> imagearray.
Also verifies the NINA-unavailable path produces a text overlay image.
"""

from __future__ import annotations

import base64
import json
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request

import numpy as np

from alpaca_camera import AlpacaDiscovery, CameraDevice, create_server

CAMERA_CFG = {
    "name": "Test Camera",
    "numx": 512,
    "numy": 512,
    "pixel_size_um": 3.75,
    "focal_length_mm": 500.0,
    "gain_default": 100,
    "gain_min": 0,
    "gain_max": 300,
    "offset_default": 50,
    "readout_modes": ["Default"],
}

RENDER_CFG = {
    "catalog": "../data/hip_catalog.csv",
    "max_mag": 12.0,
    "roll_deg": 0.0,
    "tone_gain": 2.0,
    "psf_sigma_min": 0.8,
    "psf_sigma_max": 8.0,
    "focus_ideal": 50000,
    "focus_span": 100000,
    "brightness": 40000,
    "bias": 1200,
    "simulate_noise": True,
    "noise_sigma": 40.0,
    "vignetting": 0.0,
}


class StubNina:
    def __init__(self, ra=83.82, dec=-5.3875, focus=50000):
        self.ra, self.dec, self.focus = ra, dec, focus

    def get_mount_coordinates(self):
        return self.ra, self.dec

    def get_focus_position(self):
        return self.focus


class BrokenNina:
    def get_mount_coordinates(self):
        raise RuntimeError("NINA unreachable (self-test)")

    def get_focus_position(self):
        raise RuntimeError("NINA unreachable (self-test)")


def http_get(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_put(url, form=None):
    data = urllib.parse.urlencode(form or {}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="PUT")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def decode_image(b64: str, width: int, height: int) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(b64), dtype="<u2")
    assert raw.size == width * height, f"bad size: {raw.size} != {width * height}"
    return raw.reshape(height, width)


def wait_image_ready(base, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if http_get(f"{base}/api/v1/camera/0/imageready")["Value"]:
            return True
        time.sleep(0.05)
    return False


def test_full_pipeline():
    device = CameraDevice(CAMERA_CFG, RENDER_CFG, StubNina())
    httpd = create_server("127.0.0.1", 0, device)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        info = http_get(f"{base}/management/v1/description")["Value"]
        assert info["Name"] == "Test Camera"
        devs = http_get(f"{base}/management/v1/configureddevices")["Value"]
        assert devs[0]["DeviceType"] == "camera"
        print("  management API OK")

        http_put(f"{base}/api/v1/camera/0/connected", {"Connected": "true"})
        assert http_get(f"{base}/api/v1/camera/0/connected")["Value"] is True
        assert http_get(f"{base}/api/v1/camera/0/state")["Value"] == 0
        assert http_get(f"{base}/api/v1/camera/0/sensortype")["Value"] == 0

        http_put(f"{base}/api/v1/camera/0/numx", {"NumX": "256"})
        http_put(f"{base}/api/v1/camera/0/numy", {"NumY": "256"})
        http_put(f"{base}/api/v1/camera/0/gain", {"Gain": "120"})
        assert http_get(f"{base}/api/v1/camera/0/numx")["Value"] == 256
        assert http_get(f"{base}/api/v1/camera/0/gain")["Value"] == 120
        print("  property get/set OK")

        http_put(f"{base}/api/v1/camera/0/exposurestart", {"Duration": "0.2"})
        saw_exposing = False
        for _ in range(100):
            state = http_get(f"{base}/api/v1/camera/0/state")["Value"]
            if state == 2:
                saw_exposing = True
            if state == 3:
                break
            time.sleep(0.02)
        assert saw_exposing, "camera never entered exposing state"
        print("  exposure state machine OK")

        assert wait_image_ready(base), "image never became ready"
        b64 = http_get(f"{base}/api/v1/camera/0/imagearray")["Value"]
        img = decode_image(b64, 256, 256)
        assert img.max() > 30000, "image too dark - stars not rendered"
        print(f"  exposure -> imagearray OK (max={int(img.max())})")

        dur = http_get(f"{base}/api/v1/camera/0/lastduration")["Value"]
        assert abs(dur - 0.2) < 1e-6
        print("  lastduration OK")
    finally:
        httpd.shutdown()
        thread.join(timeout=2)


def test_error_image():
    device = CameraDevice(CAMERA_CFG, RENDER_CFG, BrokenNina())
    device.start_exposure(0.0)
    deadline = time.monotonic() + 8.0
    while not device.image_ready and time.monotonic() < deadline:
        time.sleep(0.05)
    assert device.image_ready, "error image never became ready"
    img = device._last_image
    assert img.shape == (512, 512)
    assert img.max() > 10000, "error text not visible"
    assert float(np.mean(img)) < 5000, "error image should be mostly blank"
    print(f"  NINA-unavailable overlay OK (max={int(img.max())}, mean={np.mean(img):.0f})")


def test_discovery():
    discovery = AlpacaDiscovery(device_port=11111, discovery_port=0, logger=lambda *a: None)
    discovery._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    discovery._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    discovery._sock.bind(("127.0.0.1", 0))
    discovery._sock.settimeout(1.0)
    discovery._running = True
    discovery._thread = threading.Thread(target=discovery._loop, daemon=True)
    discovery._thread.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(3.0)
    try:
        port = discovery._sock.getsockname()[1]
        sock.sendto(AlpacaDiscovery.MAGIC, ("127.0.0.1", port))
        data, _ = sock.recvfrom(2048)
        resp = json.loads(data.decode("utf-8"))
        assert resp["AlpacaPort"] == 11111, resp
        assert resp["DeviceType"] == "camera", resp
        print(f"  discovery OK (response: {data.decode().strip()})")
    finally:
        sock.close()
        discovery.stop()


def main():
    print("== test: full Alpaca HTTP pipeline ==")
    test_full_pipeline()
    print("== test: NINA unavailable -> error overlay ==")
    test_error_image()
    print("== test: Alpaca UDP discovery ==")
    test_discovery()
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
