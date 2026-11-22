"""GFS point grid from NOAA's open data bucket on AWS (no key, no quota).

Writes the same ``{"pts", "data", "meta"}`` JSON as ``zjxwx.fetch`` (the
Open-Meteo downloader), so every plotting command reads it unchanged.

Why: Open-Meteo's free tier allows 10,000 weighted calls a day per IP
address, a fetch costs ~3,100, and GitHub's runners share addresses with
everyone else's jobs. On 20 and 22 Nov 2026 the briefings failed with
"Daily API request limit exceeded" before making a single request.

Source: ``s3://noaa-gfs-bdp-pds`` ``gfs.tCCz.pgrb2.0p25.fFFF`` (0.25°,
hourly to f120, 3-hourly after). Only the needed GRIB2 messages are
downloaded, using the ``.idx`` byte offsets and HTTP range requests. The
0.5° sampling grid falls exactly on 0.25° grid points, so values are taken
as they are, with no spatial interpolation.

Every field the Open-Meteo request asks for is in these files (checked
against the f000-f384 inventories, Nov 2026):

=============================  =========================================
Open-Meteo variable            GFS GRIB2 message
=============================  =========================================
precipitation (mm, past hour)  APCP:surface:0-N hour acc, differenced
cape                           CAPE:surface
lifted_index                   LFTX:surface
freezing_level_height          HGT:0C isotherm
temperature_2m                 TMP:2 m above ground (K -> °C)
temperature_{p}hPa             TMP:{p} mb (K -> °C)
relative_humidity_{p}hPa       RH:{p} mb
geopotential_height_{p}hPa     HGT:{p} mb
wind_speed/direction_{p}hPa    UGRD and VGRD:{p} mb
=============================  =========================================

Times match the Open-Meteo request: hourly from 00Z the day before
``start`` to 23Z on ``start + days``. Hours after the newest complete
cycle come from that cycle; from f120 the model is 3-hourly, so hours in
between are interpolated linearly (winds as u/v) and each 3-hour
precipitation total is spread evenly. Hours up to the newest cycle come
from the earlier cycle that covers them at f001-f006, the way Open-Meteo
keeps its past hours.

Compared with the Open-Meteo file from the same 06Z run (22 Nov 2026),
every field agrees to Open-Meteo's rounding from f049 on. Before that, and
for the past day, they differ (e.g. CAPE 200-300 J/kg lower here on
average, 250 hPa heights ~14 m higher, RH ~10 % different), because
Open-Meteo's default ``gfs_seamless`` blends a higher-resolution NCEP model
(HRRR) into the first two days over North America. This is the plain GFS.
"""
from __future__ import annotations

import http.client
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .config import CAT_LEVELS, GRID, ICING_LEVELS
from .fetch import grid_points

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
HOURLY_TO = 120   # f000-f120 hourly, then 3-hourly
LAST_FH = 384

# Open-Meteo name -> (GRIB "VAR:level" prefix, unit, conversion)
_K = (lambda v: v - 273.15)
SURFACE = {
    "cape": ("CAPE:surface", "J/kg", None),
    "lifted_index": ("LFTX:surface", "", None),
    "freezing_level_height": ("HGT:0C isotherm", "m", None),
    "temperature_2m": ("TMP:2 m above ground", "°C", _K),
}
_APCP = re.compile(r"^APCP:surface:0-(\d+) (hour|day) acc")


def _http(url, headers=None, retries=4, timeout=120):
    """GET; None on 403/404. Retries dropped connections, which cut ~1 in 3,000 range reads short."""
    for k in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}),
                                        timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if k == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            if k == retries:
                raise
        time.sleep(2 ** (k + 1))


def _messages():
    """{key: GRIB 'VAR:level' prefix} for every message downloaded per file."""
    m = {k: v[0] for k, v in SURFACE.items()}
    for p in CAT_LEVELS:
        m[f"u{p}"] = f"UGRD:{p} mb"
        m[f"v{p}"] = f"VGRD:{p} mb"
        m[f"geopotential_height_{p}hPa"] = f"HGT:{p} mb"
    for p in ICING_LEVELS:
        m[f"temperature_{p}hPa"] = f"TMP:{p} mb"
        m[f"relative_humidity_{p}hPa"] = f"RH:{p} mb"
    return m


def _url(cycle, fh):
    return f"{BUCKET}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/gfs.t{cycle:%H}z.pgrb2.0p25.f{fh:03d}"


def _fh_on_file(fh):
    return fh <= HOURLY_TO or fh % 3 == 0


def latest_cycle(need_fh, now=None, log=print):
    """Newest 6-hourly cycle whose f``need_fh`` has posted (so all earlier hours have too)."""
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    c = now.replace(hour=now.hour - now.hour % 6)
    for _ in range(8):
        fh = min(need_fh(c), LAST_FH)
        while not _fh_on_file(fh):
            fh += 1
        if _http(_url(c, fh) + ".idx", retries=1) is not None:
            return c
        log(f"GFS {c:%d/%HZ}: f{fh:03d} not posted yet")
        c -= timedelta(hours=6)
    raise RuntimeError("no complete GFS cycle on AWS in the last two days")


def plan(t0, t1, cycle):
    """{valid hour: [(cycle, fh, weight), ...]} - one entry, or two to interpolate between."""
    out = {}
    t = t0
    while t <= t1:
        if t > cycle:
            fh = int((t - cycle).total_seconds() // 3600)
            if _fh_on_file(fh):
                out[t] = [(cycle, fh, 1.0)]
            else:
                a = fh - fh % 3
                w = (fh - a) / 3
                out[t] = [(cycle, a, 1 - w), (cycle, a + 3, w)]
        else:
            # the earlier cycle that has this hour at f001-f006
            c = t - timedelta(hours=1)
            c = c.replace(hour=c.hour - c.hour % 6)
            out[t] = [(c, int((t - c).total_seconds() // 3600), 1.0)]
        t += timedelta(hours=1)
    return out


def _inventory(cycle, fh):
    raw = _http(_url(cycle, fh) + ".idx")
    if raw is None:
        return None
    rows = []
    for line in raw.decode().strip().splitlines():
        parts = line.split(":")
        rows.append((int(parts[1]), ":".join(parts[3:])))
    return rows


def _ranges(inv, fh):
    """{key: (first byte, last byte)} for the wanted messages, plus 'apcp' (0-fh total)."""
    want = _messages()
    out = {}
    for i, (off, desc) in enumerate(inv):
        end = inv[i + 1][0] - 1 if i + 1 < len(inv) else ""
        for key, pre in want.items():
            if key not in out and desc.startswith(pre + ":"):
                out[key] = (off, end)
        m = _APCP.match(desc)
        if m and "apcp" not in out:
            n = int(m.group(1)) * (24 if m.group(2) == "day" else 1)
            if n == fh:
                out["apcp"] = (off, end)
    return out


class _Points:
    """Index of the 0.5° sampling points in the 0.25° global grid."""

    def __init__(self, handle, pts):
        import eccodes

        ni, nj = eccodes.codes_get(handle, "Ni"), eccodes.codes_get(handle, "Nj")
        lat = eccodes.codes_get_array(handle, "latitudes").reshape(nj, ni)[:, 0]
        lon = eccodes.codes_get_array(handle, "longitudes").reshape(nj, ni)[0]
        lon = np.where(lon > 180, lon - 360, lon)
        pts = np.asarray(pts)
        iy = np.abs(lat[None, :] - pts[:, :1]).argmin(1)
        ix = np.abs(lon[None, :] - pts[:, 1:]).argmin(1)
        err = max(np.abs(lat[iy] - pts[:, 0]).max(), np.abs(lon[ix] - pts[:, 1]).max())
        if err > 1e-3:
            raise RuntimeError(f"GFS grid does not contain the sampling points (off by {err:.3f}°)")
        self.flat = iy * ni + ix

    def take(self, values):
        return values[self.flat]


def _download(files, workers, log):
    """{(cycle, fh): {key: values at the points}} for every file."""
    import eccodes

    from .nbm import _decode

    pts = grid_points()
    state = {"points": None}  # set by the first decoded message; every message shares the grid

    def one(cf):
        c, fh = cf
        inv = _inventory(c, fh)
        if inv is None:
            return cf, {}
        res = {}
        for key, (a, b) in _ranges(inv, fh).items():
            try:
                buf = _http(_url(c, fh), headers={"Range": f"bytes={a}-{b}"})
            except Exception as e:  # one lost message leaves that field-hour empty
                log(f"  GFS {c:%d/%HZ} f{fh:03d} {key}: {type(e).__name__}")
                continue
            if buf is None:
                continue
            h, v = _decode(buf)
            try:
                if state["points"] is None:
                    state["points"] = _Points(h, pts)
            finally:
                eccodes.codes_release(h)
            res[key] = state["points"].take(v).astype("float64")
        return cf, res

    out = {}
    with ThreadPoolExecutor(workers) as ex:
        for n, (cf, res) in enumerate(ex.map(one, sorted(files)), 1):
            out[cf] = res
            if n % 20 == 0 or n == len(files):
                log(f"  fetched {n}/{len(files)} GFS files")
    return out


def _value(data, c, fh, key):
    return data.get((c, fh), {}).get(key)


def _precip(data, c, fh):
    """Precipitation per hour in the output step ending at f``fh`` (3-hourly totals spread evenly)."""
    step = 1 if fh <= HOURLY_TO else 3
    a = _value(data, c, fh, "apcp")
    b = np.zeros_like(a) if fh - step == 0 or a is None else _value(data, c, fh - step, "apcp")
    if a is None or b is None:
        return None
    return np.clip(a - b, 0, None) / step


def fetch(start: date, days: int, step=None, workers=16, now=None, log=print):
    if step not in (None, GRID["step"]):
        raise ValueError("the AWS GFS source samples the 0.5° grid only")
    pts = grid_points()
    t0 = datetime(start.year, start.month, start.day, tzinfo=timezone.utc) - timedelta(days=1)
    t1 = datetime(start.year, start.month, start.day, 23, tzinfo=timezone.utc) + timedelta(days=days)

    cycle = latest_cycle(lambda c: int((t1 - c).total_seconds() // 3600), now=now, log=log)
    p = plan(t0, t1, cycle)
    files = {(c, fh) for parts in p.values() for c, fh, _ in parts}
    # differencing precipitation needs the previous output time too
    for c, fh in list(files):
        prev = fh - (1 if fh <= HOURLY_TO else 3)
        if prev > 0:
            files.add((c, prev))
    cycles = sorted({c for c, _ in files})
    log(f"GFS {cycle:%Y-%m-%d %HZ} (+ {len(cycles) - 1} earlier cycles for past hours): "
        f"{len(files)} files x {len(_messages()) + 1} messages from {BUCKET}")
    data = _download(files, workers, log)

    times = sorted(p)
    n = len(pts)
    names = (["precipitation"] + list(SURFACE)
             + [x for lev in CAT_LEVELS for x in (f"wind_speed_{lev}hPa", f"wind_direction_{lev}hPa",
                                                  f"geopotential_height_{lev}hPa")]
             + [x for lev in ICING_LEVELS for x in (f"temperature_{lev}hPa", f"relative_humidity_{lev}hPa")])
    conv = {k: v[2] for k, v in SURFACE.items()}
    conv.update({f"temperature_{lev}hPa": _K for lev in ICING_LEVELS})
    arr = {k: np.full((len(times), n), np.nan) for k in names}
    missing = 0

    def blend(parts, key):
        tot = None
        for c, fh, w in parts:
            v = _value(data, c, fh, key)
            if v is None:
                return None
            tot = v * w if tot is None else tot + v * w
        return tot

    for ti, t in enumerate(times):
        parts = p[t]
        # the output step that ends at or after this hour holds its precipitation
        pr = _precip(data, parts[-1][0], parts[-1][1])
        if pr is not None:
            arr["precipitation"][ti] = pr
        else:
            missing += 1
        for key in list(SURFACE) + [f"geopotential_height_{lev}hPa" for lev in CAT_LEVELS] \
                + [x for lev in ICING_LEVELS for x in (f"temperature_{lev}hPa", f"relative_humidity_{lev}hPa")]:
            v = blend(parts, key)
            if v is None:
                missing += 1
                continue
            f = conv.get(key)
            arr[key][ti] = f(v) if f else v
        for lev in CAT_LEVELS:
            u, v = blend(parts, f"u{lev}"), blend(parts, f"v{lev}")
            if u is None or v is None:
                missing += 1
                continue
            arr[f"wind_speed_{lev}hPa"][ti] = np.hypot(u, v)
            arr[f"wind_direction_{lev}hPa"][ti] = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    if missing:
        log(f"GFS: {missing} field-hours missing (left empty)")
    if all(np.isnan(a).all() for a in arr.values()):
        raise RuntimeError("GFS: nothing could be downloaded")

    units = {"time": "iso8601", "precipitation": "mm"}
    units.update({k: v[1] for k, v in SURFACE.items()})
    for lev in CAT_LEVELS:
        units.update({f"wind_speed_{lev}hPa": "m/s", f"wind_direction_{lev}hPa": "°",
                      f"geopotential_height_{lev}hPa": "m"})
    for lev in ICING_LEVELS:
        units.update({f"temperature_{lev}hPa": "°C", f"relative_humidity_{lev}hPa": "%"})
    tstr = [f"{t:%Y-%m-%dT%H:%M}" for t in times]

    def col(a, i):
        return [None if np.isnan(x) else round(float(x), 2) for x in a[:, i]]

    out = []
    for i, (la, lo) in enumerate(pts):
        out.append({"latitude": la, "longitude": lo, "hourly_units": units,
                    "hourly": {"time": tstr, **{k: col(arr[k], i) for k in names}}})
    meta = {"model": "GFS 0.25° (NOAA, AWS)", "source": BUCKET,
            "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "step": GRID["step"], "run": f"{cycle:%Y-%m-%dT%H:%MZ}"}
    return {"pts": pts, "data": out, "meta": meta}
