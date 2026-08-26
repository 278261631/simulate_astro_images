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


def _find_radec(data, ra_unit: str) -> Optional[tuple[float, float]]:
    """Recursively locate an {ra, dec} pair inside the JSON response.

    ra_unit: "hours" | "degrees" | "auto". In auto mode a plain "ra" key whose
    absolute value is <= 24 is assumed to be hours (multiplied by 15); keys with
    "deg" in the name are always degrees.
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

        ra = None
        ra_key = None
        for name in ra_names:
            if name in keys and isinstance(keys[name], (int, float)):
                ra = float(keys[name])
                ra_key = name
                break
        if ra is None:
            continue

        dec = None
        for name in dec_names:
            if name in keys and isinstance(keys[name], (int, float)):
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
            "/api/v2/mount/coordinates",
            "/api/v2/mount",
            "/api/v2/telescope/coordinates",
            "/api/v2/telescope",
        ]
        self.focus_paths = focus_paths or [
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
        """Try both data sources and report what is reachable."""
        out = {"base_url": self.base_url}
        try:
            ra, dec = self.get_mount_coordinates()
            out["mount_ra_deg"] = round(ra, 4)
            out["mount_dec_deg"] = round(dec, 4)
        except NinaError as exc:
            out["mount_error"] = str(exc)
        try:
            out["focus_position"] = self.get_focus_position()
        except NinaError as exc:
            out["focus_error"] = str(exc)
        return out
