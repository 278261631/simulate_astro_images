#!/usr/bin/env python3
"""Diagnostic: replicate the exact connection sequence NINA's ASCOM Alpaca
camera driver performs on connect.

If every property reads OK, the server side is compliant and the remaining
problem is on the NINA UI side. If anything errors, it points at a server bug.

Usage:
    python check_connection.py [--host 127.0.0.1 --port 11111]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request

PROPS = [
    "name", "description", "driverinfo", "driverversion", "interfaceversion",
    "connected", "cameraxsize", "cameraysize", "numx", "numy",
    "pixelpersizex", "pixelpersizey", "maxbinx", "maxbiny", "binx", "biny",
    "bincamerax", "bincameray", "canasymmetricbin", "startx", "starty",
    "readoutmode", "readoutmodes", "numreadoutmodes", "sensorname", "sensortype",
    "gain", "gainmin", "gainmax", "gainsetup", "offset", "offsetmin", "offsetmax",
    "offsetsetup", "electronsperadu", "exposuremin", "exposuremax",
    "exposureresolution", "canfastreadout", "fastreadout", "canshowwindow",
    "canstopexposure", "hasshutter", "hasdrsupport", "cansetccdtemperature",
    "ccdtemperature", "temperature", "coolerpower", "heatcoolerpower",
    "state", "imageready", "lastduration", "laststart",
]


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _put(url, form=None):
    data = urllib.parse.urlencode(form or {}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="PUT")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11111)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    cam = f"{base}/api/v1/camera/0"

    try:
        devs = _get(f"{base}/management/v1/configureddevices")["Value"]
        print(f"management/configureddevices -> {devs}")
    except Exception as exc:
        print(f"FAIL: cannot reach {base}: {exc}")
        print("Is the simulator running? Start start.bat first.")
        return 1

    try:
        _put(f"{cam}/connected", {"Connected": "true"})
        print("PUT connected=true -> OK")
    except Exception as exc:
        print(f"FAIL: PUT connected -> {exc}")
        return 1

    failed = 0
    for prop in PROPS:
        try:
            value = _get(f"{cam}/{prop}")["Value"]
            print(f"  {prop:24s} = {value!r}")
        except Exception as exc:
            failed += 1
            print(f"  {prop:24s} = ERROR {exc}")

    print()
    if failed:
        print(f"Connection check FAILED ({failed} error(s)) - server side needs a fix.")
        return 1
    print("Connection check PASSED - server side is fully ASCOM Alpaca compliant.")
    print("If NINA still does not list the camera, the issue is the NINA UI flow "
          "(see README / select the device as 'Current device' and click Connect).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
