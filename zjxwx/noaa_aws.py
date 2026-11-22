"""GFS or HRRR point grid from NOAA's open data buckets on AWS (no key, no quota).

Writes the ``{"pts", "data", "meta"}`` JSON every plotting command reads
(the format of the old Open-Meteo downloader, ``zjxwx.fetch``), with the
same variable names and units, so nothing downstream changes.

Why AWS: Open-Meteo's free tier allows 10,000 weighted calls a day per IP
address, a fetch costs ~3,100, and GitHub's runners share addresses with
everyone else's jobs. On 20 and 22 Nov 2026 the briefings failed with
"Daily API request limit exceeded" before making a single request.

Only the needed GRIB2 messages are downloaded, using the ``.idx`` byte
offsets and HTTP range requests, and each field is sampled at the nearest
model grid point to each 0.5° point.

=============================  =====================  ======================
grid variable                  GFS 0.25° (pgrb2)      HRRR 3 km (wrfprs)
=============================  =====================  ======================
precipitation (mm, past hour)  APCP 0-N h total,      APCP (N-1)-N h
                               differenced
cape                           CAPE:surface           CAPE:surface
lifted_index                   LFTX:surface           LFTX:500-1000 mb (*)
freezing_level_height          HGT:0C isotherm        HGT:0C isotherm
temperature_2m                 TMP:2 m above ground   TMP:2 m above ground
temperature_{p}hPa             TMP:{p} mb             TMP:{p} mb
relative_humidity_{p}hPa       RH:{p} mb              RH:{p} mb
geopotential_height_{p}hPa     HGT:{p} mb             HGT:{p} mb
wind_speed/direction_{p}hPa    UGRD/VGRD:{p} mb       UGRD/VGRD:{p} mb,
                                                      turned from grid
                                                      to true north
=============================  =====================  ======================

(*) HRRR has no ``LFTX:surface``; its 500-1000 mb lifted index is the
same parcel (a near-surface parcel lifted to 500 hPa).

Times: hourly from 00Z the day before ``start`` to 23Z on ``start + days``
(the old Open-Meteo request). Hours up to the newest complete cycle come
from the earlier run that covers them at the shortest lead (GFS f001-f006,
HRRR f01 of the hourly runs).

GFS: the newest complete 6-hourly cycle covers the rest. From f120 GFS is
3-hourly, so hours in between are interpolated linearly (winds as u/v)
and each 3-hour precipitation total is spread evenly. Checked against the
Open-Meteo file from the same 06Z run (22 Nov 2026): every field agrees to
its rounding from f049 on; before that Open-Meteo's ``gfs_seamless`` had
blended in HRRR.

HRRR: the newest 00/06/12/18Z cycle, which runs to f048. The grid ends
there, so the multi-day string outlook covers only the shifts HRRR reaches.
Its CONUS domain also misses a few sampling points in the far south-west
corner of the 0.5° grid (outside the map); those stay empty.
"""
from __future__ import annotations

import http.client
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .config import CAT_LEVELS, GRID, ICING_LEVELS
from .fetch import grid_points

_K = (lambda v: v - 273.15)


@dataclass(frozen=True)
class Model:
    name: str
    label: str
    bucket: str
    lifted_index: str       # GRIB "VAR:level" for lifted_index
    last_fh: int            # furthest forecast hour used
    hourly_to: int          # hourly output up to here, 3-hourly after
    past_lead: int          # past hours come from runs this many hours (max) before them
    cycle_every: int        # hours between the runs used for the future
    past_every: int         # hours between the runs used for past hours
    tolerance: float        # max distance (deg) from a 0.5° point to its grid point
    grid_relative_wind: bool

    def url(self, cycle, fh):
        if self.name == "hrrr":
            return f"{self.bucket}/hrrr.{cycle:%Y%m%d}/conus/hrrr.t{cycle:%H}z.wrfprsf{fh:02d}.grib2"
        return f"{self.bucket}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/gfs.t{cycle:%H}z.pgrb2.0p25.f{fh:03d}"

    def on_file(self, fh):
        return fh <= self.hourly_to or fh % 3 == 0


GFS = Model("gfs", "GFS 0.25° (NOAA, AWS)", "https://noaa-gfs-bdp-pds.s3.amazonaws.com",
            "LFTX:surface", 384, 120, 6, 6, 6, 1e-3, False)
HRRR = Model("hrrr", "HRRR 3 km (NOAA, AWS)", "https://noaa-hrrr-bdp-pds.s3.amazonaws.com",
             "LFTX:500-1000 mb", 48, 48, 1, 6, 1, 0.03, True)
MODELS = {"gfs": GFS, "hrrr": HRRR}

# HRRR's Lambert conformal grid: true at 38.5°N, centred on 97.5°W
_HRRR_CONE = np.sin(np.radians(38.5))
_HRRR_LOV = -97.5


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


def surface(model):
    """Open-Meteo name -> (GRIB "VAR:level" prefix, unit, conversion)."""
    return {
        "cape": ("CAPE:surface", "J/kg", None),
        "lifted_index": (model.lifted_index, "", None),
        "freezing_level_height": ("HGT:0C isotherm", "m", None),
        "temperature_2m": ("TMP:2 m above ground", "°C", _K),
    }


def _messages(model):
    """{key: GRIB 'VAR:level' prefix} for every message downloaded per file (besides precipitation)."""
    m = {k: v[0] for k, v in surface(model).items()}
    for p in CAT_LEVELS:
        m[f"u{p}"] = f"UGRD:{p} mb"
        m[f"v{p}"] = f"VGRD:{p} mb"
        m[f"geopotential_height_{p}hPa"] = f"HGT:{p} mb"
    for p in ICING_LEVELS:
        m[f"temperature_{p}hPa"] = f"TMP:{p} mb"
        m[f"relative_humidity_{p}hPa"] = f"RH:{p} mb"
    return m


def _floor(t, every):
    return t.replace(hour=t.hour - t.hour % every)


def latest_cycle(model, need_fh, now=None, log=print):
    """Newest run whose last needed forecast hour has posted (so all earlier hours have too)."""
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    c = _floor(now, model.cycle_every)
    for _ in range(8):
        fh = min(need_fh(c), model.last_fh)
        while not model.on_file(fh):
            fh += 1
        if _http(model.url(c, fh) + ".idx", retries=1) is not None:
            return c
        log(f"{model.name.upper()} {c:%d/%HZ}: f{fh:03d} not posted yet")
        c -= timedelta(hours=model.cycle_every)
    raise RuntimeError(f"no complete {model.name.upper()} cycle on AWS in the last two days")


def plan(model, t0, t1, cycle):
    """{valid hour: [(cycle, fh, weight), ...]} - one entry, two to interpolate between, or none."""
    out = {}
    t = t0
    while t <= t1:
        if t > cycle:
            fh = int((t - cycle).total_seconds() // 3600)
            if fh > model.last_fh:
                out[t] = []
            elif model.on_file(fh):
                out[t] = [(cycle, fh, 1.0)]
            else:
                a = fh - fh % 3
                w = (fh - a) / 3
                out[t] = [(cycle, a, 1 - w), (cycle, a + 3, w)]
        else:
            # the earlier run that has this hour at the shortest lead (>= 1 h, so it has precipitation)
            c = _floor(t - timedelta(hours=1), model.past_every)
            out[t] = [(c, int((t - c).total_seconds() // 3600), 1.0)]
        t += timedelta(hours=1)
    return out


def _inventory(model, cycle, fh):
    raw = _http(model.url(cycle, fh) + ".idx")
    if raw is None:
        return None
    rows = []
    for line in raw.decode().strip().splitlines():
        parts = line.split(":")
        rows.append((int(parts[1]), ":".join(parts[3:])))
    return rows


_APCP = re.compile(r"^APCP:surface:(\d+)-(\d+) (hour|day) acc")


def _ranges(model, inv, fh):
    """{key: (first byte, last byte)} for the wanted messages, plus 'apcp'.

    GFS: 'apcp' is the 0-fh running total (differenced later; GFS buckets
    reset every 6 h). HRRR: the (fh-1)-fh hourly amount itself.
    """
    want = _messages(model)
    out = {}
    for i, (off, desc) in enumerate(inv):
        end = inv[i + 1][0] - 1 if i + 1 < len(inv) else ""
        for key, pre in want.items():
            if key not in out and desc.startswith(pre + ":"):
                out[key] = (off, end)
        m = _APCP.match(desc)
        if m and "apcp" not in out:
            mul = 24 if m.group(3) == "day" else 1
            a, b = int(m.group(1)) * mul, int(m.group(2)) * mul
            if b == fh and a == (0 if model.name == "gfs" else fh - 1):
                out["apcp"] = (off, end)
    return out


class _Points:
    """Nearest model grid point to each 0.5° sampling point."""

    def __init__(self, handle, pts, model):
        import eccodes

        lat = eccodes.codes_get_array(handle, "latitudes")
        lon = eccodes.codes_get_array(handle, "longitudes")
        lon = np.where(lon > 180, lon - 360, lon)
        pts = np.asarray(pts, dtype=float)
        x0, x1 = pts[:, 1].min() - 1, pts[:, 1].max() + 1
        y0, y1 = pts[:, 0].min() - 1, pts[:, 0].max() + 1
        near = np.nonzero((lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1))[0]
        la, lo = lat[near], lon[near]
        idx = np.empty(len(pts), dtype=np.int64)
        err = np.empty(len(pts))
        coslat = np.cos(np.radians(pts[:, 0]))
        for i, (py, px) in enumerate(pts):
            d = (la - py) ** 2 + ((lo - px) * coslat[i]) ** 2
            k = int(d.argmin())
            idx[i] = near[k]
            err[i] = np.sqrt(d[k])
        # points off the model's grid (HRRR's CONUS domain misses the far SE corner) stay empty
        self.outside = err > model.tolerance
        if self.outside.all():
            raise RuntimeError(f"{model.name.upper()} grid does not cover the sampling points")
        self.flat = idx
        # angle from grid north to true north at each point (Lambert conformal grids)
        self.rot = np.radians(_HRRR_CONE * (lon[idx] - _HRRR_LOV)) if model.grid_relative_wind else None

    def take(self, values):
        v = values[self.flat].astype("float64")
        v[self.outside] = np.nan
        return v


def _download(model, files, workers, log):
    """({(cycle, fh): {key: values at the points}}, _Points) for every file."""
    import eccodes

    from .nbm import _decode

    pts = grid_points()
    state = {"points": None}  # set by the first decoded message; every message shares the grid

    def one(cf):
        c, fh = cf
        inv = _inventory(model, c, fh)
        if inv is None:
            return cf, {}
        res = {}
        for key, (a, b) in _ranges(model, inv, fh).items():
            try:
                buf = _http(model.url(c, fh), headers={"Range": f"bytes={a}-{b}"})
            except Exception as e:  # one lost message leaves that field-hour empty
                log(f"  {model.name.upper()} {c:%d/%HZ} f{fh:03d} {key}: {type(e).__name__}")
                continue
            if buf is None:
                continue
            h, v = _decode(buf)
            try:
                if state["points"] is None:
                    state["points"] = _Points(h, pts, model)
            finally:
                eccodes.codes_release(h)
            res[key] = state["points"].take(v)
        return cf, res

    out = {}
    with ThreadPoolExecutor(workers) as ex:
        for n, (cf, res) in enumerate(ex.map(one, sorted(files)), 1):
            out[cf] = res
            if n % 20 == 0 or n == len(files):
                log(f"  fetched {n}/{len(files)} {model.name.upper()} files")
    return out, state["points"]


def _value(data, c, fh, key):
    return data.get((c, fh), {}).get(key)


def _precip(model, data, c, fh):
    """Precipitation per hour in the output step ending at f``fh`` (3-hourly totals spread evenly)."""
    a = _value(data, c, fh, "apcp")
    if model.name == "hrrr" or a is None:
        return a
    step = 1 if fh <= model.hourly_to else 3
    b = np.zeros_like(a) if fh - step == 0 else _value(data, c, fh - step, "apcp")
    if b is None:
        return None
    return np.clip(a - b, 0, None) / step


def _files(model, p):
    files = {(c, fh) for parts in p.values() for c, fh, _ in parts}
    if model.name == "gfs":  # differencing the running total needs the previous output time too
        for c, fh in list(files):
            prev = fh - (1 if fh <= model.hourly_to else 3)
            if prev > 0:
                files.add((c, prev))
    return files


def fetch(start: date, days: int, step=None, model="gfs", workers=16, now=None, log=print):
    model = MODELS[model] if isinstance(model, str) else model
    if step not in (None, GRID["step"]):
        raise ValueError("the AWS sources sample the 0.5° grid only")
    pts = grid_points()
    t0 = datetime(start.year, start.month, start.day, tzinfo=timezone.utc) - timedelta(days=1)
    t1 = datetime(start.year, start.month, start.day, 23, tzinfo=timezone.utc) + timedelta(days=days)

    cycle = latest_cycle(model, lambda c: int((t1 - c).total_seconds() // 3600), now=now, log=log)
    p = plan(model, t0, t1, cycle)
    files = _files(model, p)
    past = sorted({c for c, _ in files} - {cycle})
    log(f"{model.name.upper()} {cycle:%Y-%m-%d %HZ} (+ {len(past)} earlier runs for past hours): "
        f"{len(files)} files x {len(_messages(model)) + 1} messages from {model.bucket}")
    uncovered = [t for t, parts in p.items() if not parts]
    if uncovered:
        log(f"{model.name.upper()} reaches f{model.last_fh:03d}: the grid ends at "
            f"{uncovered[0] - timedelta(hours=1):%Y-%m-%d %HZ}, {len(uncovered)} hours short of the request")
    data, points = _download(model, files, workers, log)
    if points is not None and points.outside.any():
        off = [pts[i] for i in np.nonzero(points.outside)[0]]
        log(f"{model.name.upper()}: {len(off)} of {len(pts)} points are off the model grid "
            f"(e.g. {off[0]}) and stay empty")

    srf = surface(model)
    # hours past the model's reach are left out, so later shifts read as "no data", not as quiet weather
    times = sorted(t for t, parts in p.items() if parts)
    n = len(pts)
    scalar = (list(srf) + [f"geopotential_height_{lev}hPa" for lev in CAT_LEVELS]
              + [x for lev in ICING_LEVELS for x in (f"temperature_{lev}hPa", f"relative_humidity_{lev}hPa")])
    names = (["precipitation"] + list(srf)
             + [x for lev in CAT_LEVELS for x in (f"wind_speed_{lev}hPa", f"wind_direction_{lev}hPa",
                                                  f"geopotential_height_{lev}hPa")]
             + [x for lev in ICING_LEVELS for x in (f"temperature_{lev}hPa", f"relative_humidity_{lev}hPa")])
    conv = {k: v[2] for k, v in srf.items()}
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
        if not parts:
            continue
        # the output step that ends at or after this hour holds its precipitation
        pr = _precip(model, data, parts[-1][0], parts[-1][1])
        if pr is not None:
            arr["precipitation"][ti] = pr
        else:
            missing += 1
        for key in scalar:
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
            if points is not None and points.rot is not None:
                a = points.rot
                u, v = np.cos(a) * u + np.sin(a) * v, -np.sin(a) * u + np.cos(a) * v
            arr[f"wind_speed_{lev}hPa"][ti] = np.hypot(u, v)
            arr[f"wind_direction_{lev}hPa"][ti] = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    if missing:
        log(f"{model.name.upper()}: {missing} field-hours missing (left empty)")
    if all(np.isnan(a).all() for a in arr.values()):
        raise RuntimeError(f"{model.name.upper()}: nothing could be downloaded")

    units = {"time": "iso8601", "precipitation": "mm"}
    units.update({k: v[1] for k, v in srf.items()})
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
    meta = {"model": model.label, "source": model.bucket,
            "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "step": GRID["step"], "run": f"{cycle:%Y-%m-%dT%H:%MZ}"}
    return {"pts": pts, "data": out, "meta": meta}
