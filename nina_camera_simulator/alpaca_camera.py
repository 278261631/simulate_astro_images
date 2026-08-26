#!/usr/bin/env python3
"""ASCOM Alpaca camera device and HTTP server for the NINA sky simulator."""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
from urllib.parse import parse_qs, urlparse

from image_renderer import (
    compute_fov_deg,
    focus_to_psf_sigma,
    get_catalog,
    render_error_image,
    render_luminance,
    to_raw16,
)

# ASCOM SensorType enum
SENSOR_TYPE = {"mono": 0, "color": 1, "rgbb": 2}


class CameraDevice:
    """State machine + rendering backend exposed as an Alpaca camera."""

    def __init__(self, cfg: dict, render_cfg: dict, nina, logger=print):
        self.cfg = cfg
        self.render_cfg = render_cfg
        self.nina = nina
        self.log = logger
        self._lock = threading.RLock()

        self.pixel_size_um = float(cfg.get("pixel_size_um", 3.75))
        self.focal_length_mm = float(cfg.get("focal_length_mm", 500.0))
        self._sensor_w = int(cfg.get("numx", 1024))
        self._sensor_h = int(cfg.get("numy", 1024))
        self.numx = self._sensor_w
        self.numy = self._sensor_h
        self.startx = 0
        self.starty = 0
        self.binx = 1
        self.biny = 1
        self.gain = float(cfg.get("gain_default", 100))
        self.offset = float(cfg.get("offset_default", 50))
        self.readout_mode = 0
        self.fast_readout = False
        self.temp_setpoint = -5.0

        self._connected = False
        self._exposing = False
        self._downloading = False
        self._image_ready = False
        self._last_image = None
        self._last_duration = 0.0
        self._last_start = None
        self._requested_duration = 0.0
        self._thread = None
        self._abort = threading.Event()
        self._focus_ideal = None  # auto-calibrated on first focuser read

        try:
            self._catalog = get_catalog(render_cfg)
        except Exception as exc:
            self._catalog = None
            self.log(f"[camera] Catalog unavailable: {exc}")

    # ---- identity (also used by the management API) ---------------------
    @property
    def name(self):
        return str(self.cfg.get("name", "NINA Simulated Alpaca Camera"))

    @property
    def unique_id(self):
        return str(uuid.uuid3(uuid.NAMESPACE_DNS, self.name + "@127.0.0.1"))

    @property
    def description(self):
        return "Alpaca camera simulator that renders a sky image from live NINA mount/focus data"

    @property
    def driver_info(self):
        return "NINA Alpaca Camera Simulator"

    @property
    def driver_version(self):
        return "1.0.0"

    @property
    def interface_version(self):
        return 3

    # ---- state -----------------------------------------------------------
    @property
    def connected(self):
        return self._connected

    @property
    def image_ready(self):
        return self._image_ready

    @property
    def state(self):
        with self._lock:
            if self._exposing:
                return 2  # exposing
            if self._downloading:
                return 3  # downloading
            return 0  # idle

    @property
    def requested_duration(self):
        return self._requested_duration

    # ---- setters ---------------------------------------------------------
    def set_connected(self, value: bool):
        with self._lock:
            self._connected = bool(value)
        if value:
            self.log("[camera] NINA connected (Alpaca camera in use)")
        else:
            self.log("[camera] NINA disconnected")
            self.abort_exposure()

    # ---- exposure --------------------------------------------------------
    def start_exposure(self, duration: float, light: bool = True):
        with self._lock:
            if self._exposing or self._downloading:
                raise RuntimeError("Exposure already in progress")
            self._requested_duration = max(0.0, float(duration))
            self._exposing = True
            self._image_ready = False
            self._last_start = datetime.now(timezone.utc).isoformat()
            self._abort.clear()
        self.log(f"[camera] Exposure started: {self._requested_duration:.3f}s")
        self._thread = threading.Thread(
            target=self._exposure_worker, args=(self._requested_duration,), daemon=True
        )
        self._thread.start()

    def abort_exposure(self):
        with self._lock:
            if not self._exposing and not self._downloading:
                return
            self._abort.set()
        self.log("[camera] Exposure abort requested")

    def _exposure_worker(self, duration: float):
        deadline = time.monotonic() + max(0.0, duration)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if self._abort.is_set():
                    with self._lock:
                        self._exposing = False
                    self.log("[camera] Exposure aborted")
                    return
                time.sleep(min(0.05, remaining))

            with self._lock:
                self._exposing = False
                self._downloading = True
            self.log("[camera] Exposure complete, capturing...")

            img = self._capture()
        except Exception as exc:
            self.log(f"[camera] Capture failed: {exc}")
            with self._lock:
                width, height = self.numx, self.numy
            img = render_error_image(width, height, "CAPTURE ERROR", str(exc))

        with self._lock:
            self._last_image = img
            self._image_ready = True
            self._last_duration = duration
            self._downloading = False
        self.log(f"[camera] Image ready: {img.shape[1]}x{img.shape[0]}")

    def _capture(self) -> "np.ndarray":
        width, height = self.numx, self.numy

        if self._catalog is None:
            raise RuntimeError("Catalog unavailable - check catalog path in config.json")

        try:
            ra, dec = self.nina.get_mount_coordinates()
        except Exception as exc:
            raise RuntimeError(f"NINA UNAVAILABLE: {exc}") from exc
        self.log(f"[camera] Mount coordinates: RA={float(ra):.4f} deg, Dec={float(dec):.4f} deg")

        focus_live = True
        try:
            focus = self.nina.get_focus_position()
            if self._focus_ideal is None:
                self._focus_ideal = focus
                self.log(f"[camera] Focuser first read - ideal focus set to {focus}")
        except Exception as exc:
            focus_live = False
            self.log(f"[camera] Focus position unavailable, using ideal focus ({exc})")
            focus = self._focus_ideal if self._focus_ideal is not None else int(
                self.render_cfg.get("focus_ideal", 0)
            )
        if focus_live:
            self.log(f"[camera] Focuser: {focus}")
        else:
            self.log(f"[camera] Focuser: {focus} (fallback - NINA focus not available)")

        rc = self.render_cfg
        fov_x, fov_y = compute_fov_deg(width, height, self.pixel_size_um, self.focal_length_mm)
        focus_ideal = self._focus_ideal if self._focus_ideal is not None else int(
            rc.get("focus_ideal", 0)
        )
        sigma = focus_to_psf_sigma(
            focus,
            focus_ideal,
            int(rc.get("focus_span", 1000)),
            float(rc.get("psf_sigma_min", 0.8)),
            float(rc.get("psf_sigma_max", 8.0)),
        )

        ra_cat, dec_cat, mag_cat = self._catalog
        lum = render_luminance(
            ra_cat, dec_cat, mag_cat,
            float(ra), float(dec),
            fov_x, fov_y, width, height,
            sigma,
            float(rc.get("tone_gain", 2.0)),
            roll_deg=float(rc.get("roll_deg", 0.0)),
        )
        img = to_raw16(
            lum,
            brightness=float(rc.get("brightness", 40000.0)),
            bias=float(rc.get("bias", 1200.0)),
            noise_sigma=float(rc.get("noise_sigma", 40.0)),
            vignetting=float(rc.get("vignetting", 0.35)),
            simulate_noise=bool(rc.get("simulate_noise", False)),
        )
        self.log(
            f"[camera] Image captured: RA={float(ra):.4f} deg, Dec={float(dec):.4f} deg, "
            f"Focuser={focus}, psf_sigma={sigma:.2f} px, fov={fov_x:.3f}x{fov_y:.3f} deg"
        )
        return img

    def image_array_jagged(self) -> list:
        """Return the 16-bit frame as a 2D jagged array ([row][col] = [Y][X]),
        matching the ASCOM Alpaca ImageArray JSON convention."""
        with self._lock:
            if self._last_image is None:
                raise RuntimeError("No image available")
            return self._last_image.tolist()

    # ---- generic Alpaca property access ----------------------------------
    def get(self, prop: str):
        prop = prop.lower()
        c = self.cfg
        with self._lock:
            if prop == "name":
                return self.name
            if prop == "description":
                return self.description
            if prop == "driverinfo":
                return self.driver_info
            if prop == "driverversion":
                return self.driver_version
            if prop == "interfaceversion":
                return self.interface_version
            if prop == "supportedactions":
                return []
            if prop == "connected":
                return self._connected
            if prop == "state":
                return self.state
            if prop == "camerastate":
                return self.state
            if prop == "cameraxsize":
                return self._sensor_w
            if prop == "cameraysize":
                return self._sensor_h
            if prop == "bitdepth":
                return 16
            if prop == "pixelsizex":
                return self.pixel_size_um
            if prop == "pixelsizey":
                return self.pixel_size_um
            if prop == "numx":
                return self.numx
            if prop == "numy":
                return self.numy
            if prop == "pixelpersizex":
                return self.pixel_size_um / 1e6
            if prop == "pixelpersizey":
                return self.pixel_size_um / 1e6
            if prop == "maxbinx":
                return 4
            if prop == "maxbiny":
                return 4
            if prop == "binx":
                return self.binx
            if prop == "biny":
                return self.biny
            if prop == "bincamerax":
                return self._sensor_w
            if prop == "bincameray":
                return self._sensor_h
            if prop == "canasymmetricbin":
                return False
            if prop == "canfastreadout":
                return True
            if prop == "fastreadout":
                return self.fast_readout
            if prop == "canstopexposure":
                return True
            if prop == "canshowwindow":
                return True
            if prop == "exposureduration":
                return self._requested_duration
            if prop == "exposuremin":
                return float(c.get("exposure_min", 0.001))
            if prop == "exposuremax":
                return float(c.get("exposure_max", 3600.0))
            if prop == "exposureresolution":
                return float(c.get("exposure_resolution", 0.001))
            if prop == "imageready":
                return self._image_ready
            if prop == "imagearray":
                return self.image_array_jagged()
            if prop == "lastduration":
                return self._last_duration
            if prop == "laststart":
                return self._last_start or ""
            if prop == "gain":
                return int(self.gain)
            if prop == "gainmin":
                return int(c.get("gain_min", 0))
            if prop == "gainmax":
                return int(c.get("gain_max", 300))
            if prop == "gainsetup":
                return True
            if prop == "gains":
                return []
            if prop == "offset":
                return int(self.offset)
            if prop == "offsetmin":
                return int(c.get("offset_min", 0))
            if prop == "offsetmax":
                return int(c.get("offset_max", 300))
            if prop == "offsetsetup":
                return True
            if prop == "electronsperadu":
                return 0.5
            if prop == "readoutmode":
                return self.readout_mode
            if prop == "readoutmodes":
                return c.get("readout_modes", ["Default"])
            if prop == "numreadoutmodes":
                return len(c.get("readout_modes", ["Default"]))
            if prop == "sensorleft":
                return self.startx
            if prop == "sensortop":
                return self.starty
            if prop == "sensorname":
                return c.get("sensorname", "NINA Sim Sensor")
            if prop == "sensortype":
                return SENSOR_TYPE.get(c.get("sensor_type", "mono"), 0)
            if prop == "bayeroffsetx":
                return 0
            if prop == "bayeroffsety":
                return 0
            if prop == "canabortexposure":
                return True
            if prop == "hasshutter":
                return False
            if prop == "isshutteropen":
                return False
            if prop == "hasdrsupport":
                return False
            if prop == "cansetccdtemperature":
                return True
            if prop == "ccdtemperature":
                return self.temp_setpoint + 0.4
            if prop == "temperature":
                return self.temp_setpoint
            if prop == "setccdtemperature":
                return self.temp_setpoint
            if prop == "cooleron":
                return self.temp_setpoint < 0
            if prop == "cangetcoolerpower":
                return True
            if prop == "coolerpower":
                return 55.0 if self.temp_setpoint < 0 else 0.0
            if prop == "heatcoolerpower":
                return 0.0
            if prop == "startx":
                return self.startx
            if prop == "starty":
                return self.starty
        raise KeyError(prop)

    def set(self, prop: str, value):
        prop = prop.lower()
        with self._lock:
            if prop == "connected":
                self._connected = self._as_bool(value)
            elif prop == "numx":
                self.numx = self._clamp_int(value, 64, self._sensor_w)
            elif prop == "numy":
                self.numy = self._clamp_int(value, 64, self._sensor_h)
            elif prop == "startx":
                self.startx = self._clamp_int(value, 0, self._sensor_w - 1)
            elif prop == "starty":
                self.starty = self._clamp_int(value, 0, self._sensor_h - 1)
            elif prop == "binx":
                self.binx = max(1, self._clamp_int(value, 1, 4))
            elif prop == "biny":
                self.biny = max(1, self._clamp_int(value, 1, 4))
            elif prop == "fastreadout":
                self.fast_readout = self._as_bool(value)
            elif prop == "exposureduration":
                self._requested_duration = float(value)
            elif prop == "gain":
                self.gain = float(value)
            elif prop == "offset":
                self.offset = float(value)
            elif prop == "readoutmode":
                self.readout_mode = int(value)
            elif prop == "temperature":
                self.temp_setpoint = float(value)
            elif prop == "setccdtemperature":
                self.temp_setpoint = float(value)
            elif prop == "cooleron":
                self.temp_setpoint = -5.0 if self._as_bool(value) else 0.0
            else:
                raise KeyError(prop)

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _clamp_int(value, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(float(value))))


class AlpacaHandler(BaseHTTPRequestHandler):
    """Handles the ASCOM Alpaca REST API for a single camera device."""

    device: CameraDevice = None
    _tids = count(1)
    server_version = "NINAAlpacaCamera/1.0.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        try:
            # args = (requestline, status_code, protocol)
            print(f"[http] {args[0]} -> {args[1] if len(args) > 1 else ''}")
        except Exception:
            pass

    @classmethod
    def _next_tid(cls):
        return next(cls._tids)

    # ---- helpers ---------------------------------------------------------
    def _parse(self):
        parts = urlparse(self.path)
        self._qs = {k: v[0] for k, v in parse_qs(parts.query).items()}
        self._segments = [s for s in parts.path.split("/") if s]
        try:
            self._cid = int(self._qs.get("ClientTransactionID", 0))
        except (TypeError, ValueError):
            self._cid = 0

    def _read_form(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
        merged = {k: v[0] for k, v in parse_qs(raw).items()}
        merged.update(self._qs)
        return merged

    @staticmethod
    def _extract_value(params: dict, prop: str):
        for key, value in params.items():
            if key.lower() == prop.lower():
                return value
        return params.get("Value")

    def _send_imagearray(self):
        """Return ImageArray with the ASCOM ArrayType/Rank fields the client needs.

        Type uses the ASCOM.Common.Alpaca.ArrayType enum: Unknown=0, Short=1,
        Int=2, Double=3. Our 16-bit pixel values are sent as Int32 (Type=2),
        matching the official ASCOM Alpaca camera simulator.
        """
        try:
            value = self.device.image_array_jagged()
        except Exception as exc:
            return self._send_alpaca_error(0, str(exc))
        sid = self._next_tid()
        payload = {
            "Value": value,
            "Type": 2,  # ArrayType.Int
            "Rank": 2,
            "ClientTransactionID": self._cid,
            "ServerTransactionID": sid,
        }
        body = json.dumps(payload).encode("utf-8")
        self._respond(200, body)

    def _send_json(self, value, cid=None, status: int = 200):
        if cid is None:
            cid = self._cid
        payload = {
            "Value": value,
            "ClientTransactionID": cid,
            "ServerTransactionID": self._next_tid(),
        }
        body = json.dumps(payload).encode("utf-8")
        self._respond(status, body)

    def _send_alpaca_error(self, errno: int, message: str, cid=None, status: int = 400):
        if cid is None:
            cid = self._cid
        payload = {
            "ErrorNumber": errno,
            "ErrorMessage": message,
            "ClientTransactionID": cid,
            "ServerTransactionID": self._next_tid(),
        }
        body = json.dumps(payload).encode("utf-8")
        self._respond(status, body)

    def _respond(self, status: int, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    # ---- management API --------------------------------------------------
    def _handle_management(self, method: str):
        device = self.device
        path = self.path.split("?", 1)[0]
        if path in ("/management/v1/description", "/management/description"):
            self._send_json({
                "ServerName": device.name,
                "Manufacturer": "simulate_astro_images",
                "ManufacturerVersion": device.driver_version,
                "Location": "Earth",
            })
        elif path in ("/management/v1/configureddevices", "/management/configureddevices"):
            self._send_json([
                {
                    "DeviceName": device.name,
                    "DeviceType": "camera",
                    "DeviceNumber": 0,
                    "UniqueID": device.unique_id,
                }
            ])
        elif path in ("/management/v1/apiversions", "/management/apiversions"):
            self._send_json([1])
        else:
            self._send_alpaca_error(0, f"Unknown management path: {path}")

    # ---- camera API ------------------------------------------------------
    def _handle_camera(self, method: str):
        device = self.device
        seg = self._segments  # ['api','v1','camera','0','<property>']
        try:
            int(seg[3])
        except (IndexError, ValueError):
            return self._send_alpaca_error(0, "Invalid device number")
        prop = seg[4] if len(seg) > 4 else None
        if not prop:
            return self._send_alpaca_error(0, "Missing property")

        if method == "GET":
            if prop.lower() == "imagearray":
                return self._send_imagearray()
            try:
                value = device.get(prop)
            except KeyError:
                return self._send_alpaca_error(0, f"Unknown property: {prop}")
            return self._send_json(value)

        params = self._read_form()
        try:
            if prop.lower() in ("startexposure", "exposurestart"):
                duration = params.get("Duration") or params.get("duration")
                if duration is None:
                    duration = device.requested_duration
                device.start_exposure(duration)
                return self._send_json(None)
            if prop.lower() in ("abortexposure", "exposureabort"):
                device.abort_exposure()
                return self._send_json(None)
            value = self._extract_value(params, prop)
            device.set(prop, value)
            return self._send_json(None)
        except Exception as exc:
            return self._send_alpaca_error(0, f"Error setting {prop}: {exc}")

    # ---- entry points ----------------------------------------------------
    def _route(self, method: str):
        path = self.path.split("?", 1)[0]
        if path.startswith("/management/"):
            self._handle_management(method)
        elif path.startswith("/api/v1/camera/"):
            self._handle_camera(method)
        else:
            self._send_alpaca_error(0, f"Unknown path: {path}")

    def do_GET(self):
        try:
            self._parse()
            self._route("GET")
        except Exception as exc:
            self._send_alpaca_error(0, f"Server error: {exc}")

    def do_PUT(self):
        try:
            self._parse()
            self._route("PUT")
        except Exception as exc:
            self._send_alpaca_error(0, f"Server error: {exc}")


def create_server(host: str, port: int, device: CameraDevice) -> ThreadingHTTPServer:
    AlpacaHandler.device = device
    httpd = ThreadingHTTPServer((host, port), AlpacaHandler)
    httpd.daemon_threads = True
    return httpd


class AlpacaDiscovery:
    """ASCOM Alpaca dynamic discovery responder (UDP port 32227).

    Clients (NINA's DeviceSelection page, the ASCOM Alpaca browser) broadcast
    the magic string "alpacadiscovery1"; a compliant device answers with a JSON
    packet {"AlpacaPort": <port>, "DeviceType": "camera"} so it shows up in the
    device browser.
    """

    MAGIC = b"alpacadiscovery1"

    def __init__(self, device_port: int, discovery_port: int = 32227,
                 device_type: str = "camera", logger=print):
        self.device_port = int(device_port)
        self.discovery_port = int(discovery_port)
        self.device_type = device_type
        self.log = logger
        self._sock = None
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind(("0.0.0.0", self.discovery_port))
        except OSError as exc:
            self.log(f"[discovery] Cannot bind UDP {self.discovery_port}: {exc}")
            self._sock.close()
            self._sock = None
            return
        self._sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.log(f"[discovery] Listening for Alpaca discovery on UDP {self.discovery_port}")

    def _loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(2048)
                if data.strip() == self.MAGIC:
                    payload = json.dumps({
                        "AlpacaPort": self.device_port,
                        "DeviceType": self.device_type,
                    }).encode("utf-8")
                    self._sock.sendto(payload, addr)
                    self.log(f"[discovery] Responded to {addr[0]}:{addr[1]}")
            except socket.timeout:
                continue
            except OSError:
                break

    def stop(self):
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
