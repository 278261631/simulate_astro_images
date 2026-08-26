#!/usr/bin/env python3
"""Entry point for the NINA Alpaca camera simulator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alpaca_camera import AlpacaDiscovery, CameraDevice, create_server
from image_renderer import compute_fov_deg
from nina_client import NinaClient


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="NINA Alpaca camera simulator")
    ap.add_argument("--config", default="config.json", help="Path to config.json")
    ap.add_argument("--host", default=None, help="Override listen host")
    ap.add_argument("--port", type=int, default=None, help="Override listen port")
    ap.add_argument("--probe", action="store_true",
                    help="Probe NINA ninaAPI 2.0 data sources and exit")
    args = ap.parse_args()

    if not Path(args.config).exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    server_cfg = cfg.get("server", {})
    nina_cfg = cfg.get("nina", {})
    camera_cfg = cfg.get("camera", {})
    render_cfg = cfg.get("render", {})

    host = args.host or server_cfg.get("host", "127.0.0.1")
    port = args.port or int(server_cfg.get("port", 11111))

    nina = NinaClient(
        base_url=nina_cfg.get("base_url", "http://127.0.0.1:4557"),
        api_key=nina_cfg.get("api_key", ""),
        timeout=float(nina_cfg.get("timeout_seconds", 3.0)),
        mount_paths=[nina_cfg.get("mount_coordinates_path", "/api/v2/mount/coordinates")]
        + list(nina_cfg.get("mount_fallback_paths", [])),
        focus_paths=[nina_cfg.get("focus_position_path", "/api/v2/focuser/position")]
        + list(nina_cfg.get("focus_fallback_paths", [])),
        ra_unit=nina_cfg.get("ra_unit", "auto"),
    )

    if args.probe:
        print(f"Probing NINA ninaAPI 2.0 at {nina.base_url} ...")
        for key, value in nina.probe().items():
            print(f"  {key}: {value}")
        return 0

    device = CameraDevice(camera_cfg, render_cfg, nina)
    fov_x, fov_y = compute_fov_deg(device.numx, device.numy,
                                   device.pixel_size_um, device.focal_length_mm)

    print("=" * 62)
    print("NINA Alpaca camera simulator")
    print(f"  Sensor     : {device.numx}x{device.numy} @ {device.pixel_size_um}um, "
          f"focal {device.focal_length_mm}mm")
    print(f"  Field of view: {fov_x:.3f} x {fov_y:.3f} deg")
    print(f"  Alpaca URL : http://{host}:{port}/api/v1/camera/0")
    print(f"  NINA source: {nina.base_url} (ninaAPI 2.0)")
    if server_cfg.get("discovery_enabled", True):
        print(f"  Discovery  : UDP {server_cfg.get('discovery_port', 32227)} "
              "(NINA DeviceSelection will list this camera)")
    print("  In NINA: Equipment > Camera > Alpaca Camera -> use the Alpaca URL above.")
    print("=" * 62)

    try:
        httpd = create_server(host, port, device)
    except OSError as exc:
        print(f"Failed to bind {host}:{port}: {exc}", file=sys.stderr)
        return 1

    discovery = None
    if server_cfg.get("discovery_enabled", True):
        discovery = AlpacaDiscovery(
            device_port=port,
            discovery_port=int(server_cfg.get("discovery_port", 32227)),
            device_type="camera",
        )
        discovery.start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        if discovery is not None:
            discovery.stop()
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
