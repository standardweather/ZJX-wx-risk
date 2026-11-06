"""National Blend of Models (NBM) fields from NOAA's open data bucket on AWS.

NBM v4 is published hourly on ``s3://noaa-nbm-grib2-pds`` (no key, no
quota). Each CONUS file is ~160 MB, so only the needed GRIB2 messages are
downloaded, using the ``.idx`` byte offsets and HTTP range requests, and
the 2.5 km grid is cut to the ZJX map extent and thinned to ~5 km.

What NBM covers for the OM/TMU/OS maps (checked against the v4.3 CONUS
``core`` inventory, Nov 2026):

* thunder: 1-h thunderstorm probability (TSTM, %) every hour to f036
* IFR/LIFR: P(ceiling < 1000 / 500 ft) and P(visibility < 3 / 1 SM).
  NBM leaves these undefined over water, so they are land-only.
* low-level wind shear: LLWS height (DIST, surface-610 m AGL) and the wind
  at that height. Zero where NBM does not diagnose LLWS.
* freezing rain: 1-h ice accretion (FICEAC).

NBM does **not** carry in-flight icing aloft, and its only upper-level
turbulence field is an undocumented 0-3 Ellrod category for 400-300 hPa.
Those hazards therefore come from GFS (see ``tmu.py``). NBM's low-level
turbulence category (TURB, surface-1372 m AGL) is not used either: on the
6 Nov 2026 run it put "severe" and "extreme" over 6 % of ZJX where
10 m winds were ~8 kt and gusts ~13 kt, so its scale cannot be trusted
without verification.
"""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from .config import EXTENT

BUCKET = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
MAX_HOURLY_FH = 36  # every NBM cycle has hourly core output to f036

# name -> (idx description prefix, accumulation?)  {f} = forecast hour, {p} = f-1
FIELDS = {
    "tstm": ("TSTM:surface:{p}-{f} hour acc fcst:probability forecast", True),
    "fzra": ("FICEAC:surface:{p}-{f} hour acc fcst:", True),
    "ceil_lt1000": ("CEIL:cloud ceiling:{f} hour fcst:prob <304.8:", False),
    "ceil_lt500": ("CEIL:cloud ceiling:{f} hour fcst:prob <152.4:", False),
    "vis_lt3": ("VIS:surface:{f} hour fcst:prob <4828.03:", False),
    "vis_lt1": ("VIS:surface:{f} hour fcst:prob <1609.34:", False),
    "llws_hgt": ("DIST:surface - 610 m above ground:{f} hour fcst:", False),
    "llws_spd": ("WIND:surface - 610 m above ground:{f} hour fcst:", False),
}


@dataclass
class NbmGrid:
    lats: np.ndarray                # (ny, nx)
    lons: np.ndarray                # (ny, nx)
    times: np.ndarray               # (nt,) datetime64[m] UTC valid times
    fields: dict = field(repr=False)  # name -> (nt, ny, nx) float32, NaN = missing
    meta: dict = field(default_factory=dict)

    def get(self, name):
        return self.fields.get(name)

    def index(self, when: datetime):
        t = np.datetime64(when.astimezone(timezone.utc).replace(tzinfo=None), "m")
        hits = np.nonzero(self.times == t)[0]
        return int(hits[0]) if hits.size else None

    def save(self, path):
        np.savez_compressed(path, lats=self.lats, lons=self.lons, times=self.times.astype("int64"),
                            meta=np.array(repr(self.meta)),
                            **{f"f_{k}": v for k, v in self.fields.items()})

    @classmethod
    def load(cls, path):
        import ast

        z = np.load(path)
        fields = {k[2:]: z[k] for k in z.files if k.startswith("f_")}
        return cls(z["lats"], z["lons"], z["times"].astype("datetime64[m]"), fields,
                   ast.literal_eval(str(z["meta"])))

    @property
    def label(self):
        return f"{self.meta.get('model', 'NBM v4')} {self.meta.get('cycle', '')}".strip()


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
def _http(url, headers=None, retries=4, timeout=120):
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
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if k == retries:
                raise
        time.sleep(2 ** (k + 1))


def _url(cycle: datetime, fh: int):
    return (f"{BUCKET}/blend.{cycle:%Y%m%d}/{cycle:%H}/core/"
            f"blend.t{cycle:%H}z.core.f{fh:03d}.co.grib2")


def _inventory(cycle, fh):
    raw = _http(_url(cycle, fh) + ".idx")
    if raw is None:
        return None
    rows = []
    for line in raw.decode().strip().splitlines():
        parts = line.split(":")
        rows.append((int(parts[1]), ":".join(parts[3:])))
    return rows


def pick_cycle(t0: datetime, t1: datetime, now=None, log=print):
    """Newest NBM cycle that covers [t0, t1] with hourly output.

    Prefers a cycle issued before ``t0`` (so the first hours of the first
    period are true forecasts), then falls back to the newest cycle that
    reaches ``t1`` at all.
    """
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    fallback = None
    for back in range(0, 30):
        c = now - timedelta(hours=back)
        last = int((t1 - c).total_seconds() // 3600)
        if last > MAX_HOURLY_FH:
            continue
        if _http(_url(c, max(last, 1)) + ".idx", retries=1) is None:
            continue  # this cycle has not finished posting yet
        if c < t0:
            return c
        fallback = fallback or c
    if fallback:
        log(f"NBM: no cycle before {t0:%d/%H%MZ} reaches {t1:%d/%H%MZ}; using {fallback:%d/%HZ}")
        return fallback
    raise RuntimeError(f"no NBM cycle covers {t0:%Y-%m-%d %H%MZ} to {t1:%Y-%m-%d %H%MZ}")


# --------------------------------------------------------------------------
# Download and decode
# --------------------------------------------------------------------------
class _Subset:
    """Row/column window of the NBM CONUS grid covering the map extent."""

    def __init__(self, handle, stride, pad=1.0):
        import eccodes

        ni, nj = eccodes.codes_get(handle, "Nx"), eccodes.codes_get(handle, "Ny")
        lat = eccodes.codes_get_array(handle, "latitudes").reshape(nj, ni)
        lon = eccodes.codes_get_array(handle, "longitudes").reshape(nj, ni)
        lon = np.where(lon > 180, lon - 360, lon)
        x0, x1, y0, y1 = EXTENT
        inside = (lon >= x0 - pad) & (lon <= x1 + pad) & (lat >= y0 - pad) & (lat <= y1 + pad)
        rows, cols = np.nonzero(inside)
        self.shape = (nj, ni)
        self.sl = (slice(rows.min(), rows.max() + 1, stride), slice(cols.min(), cols.max() + 1, stride))
        self.lats, self.lons = lat[self.sl].astype("float32"), lon[self.sl].astype("float32")

    def cut(self, values):
        return values.reshape(self.shape)[self.sl]


def _decode(buf):
    import eccodes

    h = eccodes.codes_new_from_message(buf)
    try:
        v = eccodes.codes_get_values(h)
        miss = eccodes.codes_get(h, "missingValue")
        v = np.where(v == miss, np.nan, v)
        return h, v
    except Exception:
        eccodes.codes_release(h)
        raise


def fetch(t0: datetime, t1: datetime, stride=2, workers=8, cycle=None, log=print) -> NbmGrid:
    """Hourly NBM fields valid from t0 to t1 (UTC, inclusive) over the map extent."""
    import eccodes

    cycle = cycle or pick_cycle(t0, t1, log=log)
    first = max(1, int((t0 - cycle).total_seconds() // 3600))
    last = int((t1 - cycle).total_seconds() // 3600)
    hours = list(range(first, last + 1))
    log(f"NBM {cycle:%Y-%m-%d %HZ}: f{first:03d}-f{last:03d}, {len(FIELDS)} fields")

    def plan(fh):
        inv = _inventory(cycle, fh)
        if inv is None:
            return fh, {}
        out = {}
        for name, (pattern, _) in FIELDS.items():
            pre = pattern.format(f=fh, p=fh - 1)
            for i, (off, desc) in enumerate(inv):
                if desc.startswith(pre):
                    end = inv[i + 1][0] - 1 if i + 1 < len(inv) else ""
                    out[name] = (off, end)
                    break
        return fh, out

    with ThreadPoolExecutor(workers) as ex:
        plans = dict(ex.map(plan, hours))

    jobs = [(fh, name, rng) for fh in hours for name, rng in plans[fh].items()]

    def grab(job):
        fh, name, (a, b) = job
        return fh, name, _http(_url(cycle, fh), headers={"Range": f"bytes={a}-{b}"})

    with ThreadPoolExecutor(workers) as ex:
        blobs = list(ex.map(grab, jobs))

    sub, data, version = None, {}, None
    for fh, name, buf in blobs:
        if buf is None:
            continue
        h, v = _decode(buf)
        if sub is None:
            sub = _Subset(h, stride)
            try:
                version = eccodes.codes_get(h, "generatingProcessIdentifier")
            except Exception:
                version = None
        eccodes.codes_release(h)
        data[(fh, name)] = sub.cut(v).astype("float32")
    if sub is None:
        raise RuntimeError(f"NBM {cycle:%Y-%m-%d %HZ}: no fields could be downloaded")

    nt = len(hours)
    fields = {n: np.full((nt,) + sub.lats.shape, np.nan, dtype="float32") for n in FIELDS}
    for (fh, name), arr in data.items():
        fields[name][hours.index(fh)] = arr
    missing = [f"{n} f{fh:03d}" for fh in hours for n in FIELDS if (fh, n) not in data]
    if missing:
        log(f"NBM: {len(missing)} messages missing, e.g. {missing[:4]}")
    times = np.array([np.datetime64((cycle + timedelta(hours=fh)).replace(tzinfo=None), "m")
                      for fh in hours])
    meta = {"model": "NBM v4", "cycle": f"{cycle:%Y-%m-%d %HZ}", "source": BUCKET, "stride_km": 2.5 * stride,
            "missing": len(missing)}
    if version is not None:
        meta["process"] = int(version)
    return NbmGrid(sub.lats, sub.lons, times, fields, meta)


_CYCLE_RE = re.compile(r"(\d{8})(\d{2})")


def parse_cycle(s):
    """'2026110604' or '2026-11-06T04' -> aware UTC datetime."""
    m = _CYCLE_RE.search(re.sub(r"[-T:Z ]", "", s))
    if not m:
        raise ValueError(f"cycle must look like YYYYMMDDHH, got {s!r}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)
