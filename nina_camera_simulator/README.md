# NINA Alpaca Camera Simulator

An ASCOM **Alpaca camera** simulator that renders a sky image from live data read
out of NINA via the **ninaAPI 2.0** plugin. The mount's current RA/Dec sets the
pointing direction, and the focuser position controls defocus (PSF blur) in the
rendered frame.

```
NINA (ninaAPI 2.0) ──► nina_client.py ──► CameraDevice ──► Alpaca HTTP server
     mount RA/Dec,                                   │
     focuser position                               ▼
                                        image_renderer.py (Hipparcos catalog)
```

The star-field rendering reuses the projection / PSF kernels from
`../python/render_sky_patch.py`, so results stay consistent with the rest of the
repo.

## Layout

| File              | Purpose                                                      |
|-------------------|--------------------------------------------------------------|
| `server.py`       | Entry point (`--probe` to test the NINA connection)          |
| `alpaca_camera.py`| Alpaca camera device + HTTP server (stdlib `http.server`)    |
| `nina_client.py`  | Minimal ninaAPI 2.0 client (mount coords, focuser position)  |
| `image_renderer.py`| Star-field rendering, 16-bit frame build, error overlay     |
| `config.json`     | Server / NINA / camera / rendering settings                  |
| `selftest.py`     | End-to-end self test with a stub NINA (no NINA required)     |

## Requirements

```
pip install -r requirements.txt        # numpy, matplotlib, Pillow
```

## Usage

```powershell
# 1. Check NINA is reachable and print what we can read from it
python server.py --probe

# 2. Start the Alpaca camera server
python server.py                       # uses config.json
python server.py --port 11112          # override

# 3. In NINA:
#    Equipment > Camera > add device type "Alpaca Camera"
#    - Browse (DeviceSelection) should now list "NINA Simulated Alpaca Camera"
#      automatically (UDP discovery on port 32227), or
#    - enter the address manually:  http://127.0.0.1:11111/api/v1/camera/0
#    Connect and expose.
```

The simulator responds to the ASCOM Alpaca dynamic discovery protocol
(`alpacadiscovery1` on UDP port 32227), so the camera shows up in NINA's
DeviceSelection browser at `http://localhost:63960/DeviceSelection`. Disable
discovery in `config.json` if the UDP port conflicts with something else.

When you start an exposure, the simulator queries NINA for the current mount
RA/Dec and focuser position, renders a frame, and marks it ready. NINA then
downloads the 16-bit frame.

**If NINA is not reachable**, the camera still produces a (mostly blank) frame
with the error text overlaid on it, so NINA does not hang and the cause is
visible on the image.

## config.json

- `server.host` / `server.port` – listen address of the Alpaca HTTP server.
- `server.discovery_enabled` / `server.discovery_port` – ASCOM Alpaca dynamic
  discovery (UDP 32227) so the device appears in NINA's DeviceSelection browser.
- `nina.base_url` – ninaAPI 2.0 plugin address. The default is
  `http://127.0.0.1:4557`, but the plugin's port is configurable in NINA
  (Options > Advanced API); set this to match your actual port
  (e.g. `http://127.0.0.1:1888`).
- `nina.mount_coordinates_path` / `focus_position_path` – primary endpoints.
  ninaAPI 2.0 mounts under `/v2/api`, so the defaults are
  `/v2/api/equipment/mount/info` and `/v2/api/equipment/focuser/info`; the
  fallback lists cover older route layouts and are tried automatically.
- `nina.ra_unit` – `auto` (a plain `ra` <= 24 is treated as hours), `hours`, or
  `degrees`.
- `nina.api_key` – optional token sent as `X-Api-Key` if the plugin requires it.
- `camera.*` – sensor size, pixel size, focal length (defines the field of view),
  gain/offset/exposure limits, readout modes.
- `render.*` – catalog path, magnitude limit, roll, tone-mapping gain, and the
  **focus → PSF blur** mapping (`focus_ideal` = sharpest position, `focus_span` =
  defocus distance at which blur is maximum).

## Notes / assumptions

- **ImageArray format**: the 16-bit frame is returned as a 2D jagged integer
  array in the JSON `Value` field (`[[row0...],[row1...]]`, i.e. `[Y][X]`), which
  is the format the ASCOM Alpaca camera client (and NINA) expects.
- **RA unit heuristic**: ninaAPI responses historically return RA in hours. In
  `auto` mode a plain `ra` key whose value is <= 24 is multiplied by 15.
  Override with `nina.ra_unit` if your setup differs.
- The catalog is loaded once and cached per (path, max magnitude).

## Troubleshooting

**Device is visible in the DeviceSelection page but not in NINA.**

The server side is almost certainly fine — run `python check_connection.py` to
replicate NINA's full connection sequence. If it passes, complete the NINA UI
flow:

1. Equipment > Camera dropdown > "Alpaca Camera".
2. Click **Browse** (opens `http://localhost:63960/DeviceSelection`).
3. **Click the device row** `127.0.0.1:11111 (Alpaca Camera 0)` so that
   "Current device" becomes filled in (it is empty by default).
4. Close the page, back in the NINA dialog the host/port are now filled.
5. Click **Connect / OK**. The camera now appears in the dropdown.

If NINA previously stored a "Manually configured device", it may already be a
selectable entry in the camera dropdown — just select it and connect.

## Testing without NINA

```powershell
python selftest.py        # or: test_smoke.bat
```

Exercises the full Alpaca pipeline (management API, connect, property get/set,
exposure state machine, image download) using a stub NINA, and verifies the
error-overlay path when NINA is unavailable.
