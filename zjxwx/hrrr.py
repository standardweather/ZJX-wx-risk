"""HRRR convection fields from NOAA's open data bucket on AWS.

Why HRRR: on 6 Nov 2026 high-topped storms (MRMS echo tops 42-49 kft)
crossed South Carolina from 10Z. NBM 05Z gave 0 % thunder there and the
GFS signal was too weak for its thresholds, but HRRR 00Z and 04Z both
forecast 35-42 kft tops with lightning in the right place. HRRR is a
3 km convection-allowing model, so it is the best free source for where
storms will actually be in the first day.

Fields (``wrfsfc`` files, hourly): simulated echo top (RETOP, m) and
lightning threat (LTNG, flashes km^-2 per 5 min). Every hourly cycle runs
to f018; the 00/06/12/18Z cycles run to f048. For each valid hour the
newest cycle that covers it is used, so the morning run blends the 04Z
cycle (to ~22Z) with the 00Z extended cycle (overnight).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np

from .nbm import NbmGrid, _decode, _http, _Subset

BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
FIELDS = {"retop": "RETOP:cloud top:{f} hour fcst:", "ltng": "LTNG:entire atmosphere:{f} hour fcst:"}


def _url(cycle, fh):
    return f"{BUCKET}/hrrr.{cycle:%Y%m%d}/conus/hrrr.t{cycle:%H}z.wrfsfcf{fh:02d}.grib2"


def _reach(cycle):
    return 48 if cycle.hour % 6 == 0 else 18


def plan(t0: datetime, t1: datetime, now=None, log=print):
    """{valid time: (cycle, fh)} using the newest posted cycle for each hour."""
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    hours = []
    t = t0
    while t <= t1:
        hours.append(t)
        t += timedelta(hours=1)
    posted = {}

    def ok(c, fh):
        key = (c, fh)
        if key not in posted:
            posted[key] = _http(_url(c, fh) + ".idx", retries=1) is not None
        return posted[key]

    out = {}
    for back in range(0, 30):
        c = now - timedelta(hours=back)
        need = [h for h in hours if h not in out and 1 <= (h - c).total_seconds() / 3600 <= _reach(c)]
        if not need:
            continue
        last = int((max(need) - c).total_seconds() // 3600)
        if not ok(c, last):
            continue  # not (fully) posted yet
        for h in need:
            out[h] = (c, int((h - c).total_seconds() // 3600))
        if len(out) == len(hours):
            break
    missing = [h for h in hours if h not in out]
    if missing:
        log(f"HRRR: no cycle covers {len(missing)} hour(s), e.g. {missing[0]:%d/%H%MZ}")
    if not out:
        raise RuntimeError("no HRRR cycle covers the requested period")
    return out


def fetch(t0, t1, stride=2, workers=8, now=None, log=print) -> NbmGrid:
    import eccodes

    p = plan(t0, t1, now, log)
    cycles = sorted({c for c, _ in p.values()})
    log("HRRR cycles: " + ", ".join(f"{c:%d/%HZ}" for c in cycles))

    def grab(item):
        valid, (c, fh) = item
        raw = _http(_url(c, fh) + ".idx")
        if raw is None:
            return valid, {}
        inv = [(int(x.split(":")[1]), ":".join(x.split(":")[3:])) for x in raw.decode().strip().splitlines()]
        res = {}
        for name, pat in FIELDS.items():
            pre = pat.format(f=fh)
            for i, (off, desc) in enumerate(inv):
                if desc.startswith(pre):
                    end = inv[i + 1][0] - 1 if i + 1 < len(inv) else ""
                    res[name] = _http(_url(c, fh), headers={"Range": f"bytes={off}-{end}"})
                    break
        return valid, res

    with ThreadPoolExecutor(workers) as ex:
        blobs = list(ex.map(grab, sorted(p.items())))

    times = sorted(p)
    sub, fields = None, None
    for valid, res in blobs:
        for name, buf in res.items():
            if buf is None:
                continue
            h, v = _decode(buf)
            if sub is None:
                sub = _Subset(h, stride)
                fields = {n: np.full((len(times),) + sub.lats.shape, np.nan, "float32") for n in FIELDS}
            eccodes.codes_release(h)
            fields[name][times.index(valid)] = sub.cut(v)
    if sub is None:
        raise RuntimeError("HRRR: no fields could be downloaded")
    tarr = np.array([np.datetime64(t.replace(tzinfo=None), "m") for t in times])
    meta = {"model": "HRRR", "cycle": f"{cycles[0]:%Y-%m-%d} " + "/".join(f"{c:%HZ}" if c.date() == cycles[0].date() else f"{c:%d/%HZ}" for c in cycles), "source": BUCKET,
            "stride_km": 3 * stride}
    return NbmGrid(sub.lats, sub.lons, tarr, fields, meta)
