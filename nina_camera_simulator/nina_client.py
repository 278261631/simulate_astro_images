#!/usr/bin/env python3
"""Minimal ninaAPI 2.0 client: reads mount RA/Dec and focuser position from NINA."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional


class NinaError(RuntimeError):
    """Raised when NINA's ninaAPI 2.0 plugin cannot be reached or returns bad data."""


def _iter_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_dicts(item)


def _num(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _angle_field(angle, *fields):
    """Extract a numeric field from a NINA Angle dict (has Hours/Degree/Radians)."""
    if isinstance(angle, dict):
        low = {str(k).lower(): v for k, v in angle.items()}
        for field in fields:
            value = _num(low.get(field))
            if value is not None:
                return value
    return None


def _find_radec(data, ra_unit: str) -> Optional[tuple[float, float]]:
    """Recursively locate the mount's current RA/Dec inside the JSON response.

    Understands the shapes returned by ninaAPI 2.0:
      - NINA TelescopeInfo: top-level "RightAscension" (hours) + "Declination"
        (degrees), plus a "Coordinates" object with nested Angle dicts
        {Radians, Hours, Degree, ...}.
      - Simple objects like {"ra": ..., "dec": ...}.

    ra_unit: "hours" | "degrees" | "auto". In auto mode a plain "ra" whose
    absolute value is <= 24 is assumed to be hours (x15); keys containing "deg"
    are always degrees. NINA's "RightAscension"/"Coordinates.RA.Hours" are
    always treated as hours.
    """
    ra_names = (
        "ra", "rightascension", "ra_deg", "radegrees", "ra_hours", "rahours",
        "rah", "ra_degrees", "right_ascension",
    )
    dec_names = (
        "dec", "declination", "dec_deg", "decdegrees", "declinationdegrees",
        "declination_degrees", "dech", "dec_hours",
    )

    for d in _iter_dicts(data):
        keys = {str(k).lower(): v for k, v in d.items()}

        # 1) NINA Coordinates object: {"Coordinates": {"RA": {...}, "Dec": {...}}}
        coords = keys.get("coordinates")
        if isinstance(coords, dict):
            ck = {str(k).lower(): v for k, v in coords.items()}
            ra_hours = _angle_field(ck.get("ra"), "hours", "degree", "radians")
            dec_deg = _angle_field(ck.get("dec"), "degree", "radians")
            if ra_hours is not None and dec_deg is not None:
                return ra_hours * 15.0, dec_deg

        # 2) NINA TelescopeInfo top-level: RightAscension (hours) + Declination
        ra = _num(keys.get("rightascension"))
        dec = _num(keys.get("declination"))
        if ra is not None and dec is not None:
            return ra * 15.0, dec

        # 3) plain ra/dec keys
        ra = None
        ra_key = None
        for name in ra_names:
            if name in keys and _num(keys[name]) is not None:
                ra = float(keys[name])
                ra_key = name
                break
        if ra is None:
            continue
        dec = None
        for name in dec_names:
            if name in keys and _num(keys[name]) is not None:
                dec = float(keys[name])
                break
        if dec is None:
            continue

        if ra_unit == "hours":
            ra *= 15.0
        elif ra_unit == "degrees":
            pass
        else:  # auto
            if abs(ra) <= 24.0 and "deg" not in ra_key:
                ra *= 15.0
        return ra, dec
    return None


def _find_focus(data) -> Optional[int]:
    preferred = (
        "position", "focuserposition", "focusposition", "step", "steps", "value",
    )
    for d in _iter_dicts(data):
        keys = {str(k).lower(): v for k, v in d.items()}
        for name in preferred:
            value = keys.get(name)
            if isinstance(value, (int, float)):
                return int(value)
    return None


class NinaClient:
    """Tiny HTTP client for the ninaAPI 2.0 plugin.

    Tries a configurable list of candidate paths per data source and picks the
    first one that returns parseable JSON.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:4557",
        api_key: str = "",
        timeout: float = 3.0,
        mount_paths: Optional[list[str]] = None,
        focus_paths: Optional[list[str]] = None,
        ra_unit: str = "auto",
    ):
        self.base_url = (base_url or "http://127.0.0.1:4557").rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.mount_paths = mount_paths or [
            "/v2/api/equipment/mount/info",
            "/v2/api/equipment/telescope/info",
            "/api/v2/mount/coordinates",
            "/api/v2/mount",
        ]
        self.focus_paths = focus_paths or [
            "/v2/api/equipment/focuser/info",
            "/api/v2/focuser/position",
            "/api/v2/focuser",
        ]
        self.ra_unit = ra_unit

    def _request(self, path: str):
        url = self.base_url + path
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        if self.api_key:
            req.add_header("X-Api-Key", self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
            if not raw.strip():
                raise NinaError(f"Empty response from {url}")
            return json.loads(raw)
        except NinaError:
            raise
        except urllib.error.HTTPError as exc:
            raise NinaError(f"HTTP {exc.code} from {url}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise NinaError(f"Cannot reach {url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise NinaError(f"Invalid JSON from {url}: {exc}") from exc

    def _first_ok(self, paths: list[str]):
        errors = []
        for path in paths:
            try:
                return self._request(path)
            except NinaError as exc:
                errors.append(str(exc))
        raise NinaError(" | ".join(errors))

    def get_mount_coordinates(self) -> tuple[float, float]:
        """Return (ra_deg, dec_deg) of the mount's current pointing direction."""
        data = self._first_ok(self.mount_paths)
        found = _find_radec(data, self.ra_unit)
        if found is None:
            raise NinaError(f"No RA/Dec found in response: {data!r}")
        return found

    def get_focus_position(self) -> int:
        """Return the current focuser step position."""
        data = self._first_ok(self.focus_paths)
        found = _find_focus(data)
        if found is None:
            raise NinaError(f"No focuser position found in response: {data!r}")
        return found

    def probe(self) -> dict:
        """Try both data sources, report reachability + parsed values + raw JSON."""
        out = {"base_url": self.base_url, "mount_paths": list(self.mount_paths),
               "focus_paths": list(self.focus_paths)}
        try:
            data = self._first_ok(self.mount_paths)
            out["mount_raw"] = data
            ra, dec = _find_radec(data, self.ra_unit)
            if ra is None:
                out["mount_error"] = f"No RA/Dec found in response: {data!r}"
            else:
                out["mount_ra_deg"] = round(ra, 4)
                out["mount_dec_deg"] = round(dec, 4)
        except NinaError as exc:
            out["mount_error"] = str(exc)
        try:
            data = self._first_ok(self.focus_paths)
            out["focus_raw"] = data
            focus = _find_focus(data)
            if focus is None:
                out["focus_error"] = f"No focuser position found in response: {data!r}"
            else:
                out["focus_position"] = focus
        except NinaError as exc:
            out["focus_error"] = str(exc)
        return out
