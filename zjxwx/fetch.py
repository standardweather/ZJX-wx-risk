"""Download a GFS point grid from the Open-Meteo GFS API.

Writes the same ``{"pts", "data", "meta"}`` JSON the plotting commands read.
Requests are batched (many coordinates per call), paced to stay inside
Open-Meteo's per-minute limit, and retried with backoff. The first day
requested is the day *before* the briefing so the Ellrod-Knox divergence
trend has a 6-hour look-back at the start of a shift.

Open-Meteo counts "weighted" calls: roughly one per location per 10
variables (and per 2 weeks of data). The free tier allows 600/min,
5,000/hour and 10,000/day for non-commercial use; set OPEN_METEO_API_KEY
(customer API) for commercial use and higher limits.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .config import CAT_LEVELS, GRID, ICING_LEVELS

FREE_API = "https://api.open-meteo.com/v1/gfs"
CUSTOMER_API = "https://customer-api.open-meteo.com/v1/gfs"
# convective_inhibition is optional in diagnostics; Open-Meteo serves it only
# for some GFS variants, so it is left out to keep the request at 40 variables.
SURFACE = ["precipitation", "cape", "lifted_index", "freezing_level_height", "temperature_2m"]
PER_MINUTE = 500  # stay under the 600/min free-tier limit


def variables():
    v = list(SURFACE)
    for p in CAT_LEVELS:
        v += [f"wind_speed_{p}hPa", f"wind_direction_{p}hPa", f"geopotential_height_{p}hPa"]
    for p in ICING_LEVELS:
        v += [f"temperature_{p}hPa", f"relative_humidity_{p}hPa"]
    return v


def grid_points(step=None):
    step = step or GRID["step"]
    lats = np.arange(GRID["lat0"], GRID["lat1"] + 1e-9, step)
    lons = np.arange(GRID["lon0"], GRID["lon1"] + 1e-9, step)
    return [[round(float(a), 3), round(float(o), 3)] for a in lats for o in lons]


class _BadVariable(Exception):
    pass


def _get(params, api, retries=5, log=print):
    url = api + "?" + urllib.parse.urlencode(params, safe=",")
    for k in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if e.code == 400:
                reason = _reason(body)
                raise _BadVariable(reason)
            wait = 65 if e.code == 429 else 2 ** (k + 1)
            log(f"  HTTP {e.code} ({_reason(body)}); retry in {wait}s")
            if k == retries:
                raise
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            if k == retries:
                raise
            log(f"  network error {e}; retry in {2 ** (k + 1)}s")
            time.sleep(2 ** (k + 1))


def _reason(body):
    try:
        return json.loads(body).get("reason", body[:200])
    except ValueError:
        return body[:200]


def fetch(start: date, days: int, step=None, batch=100, model=None, log=print):
    pts = grid_points(step)
    names = variables()
    key = os.environ.get("OPEN_METEO_API_KEY")
    api = CUSTOMER_API if key else FREE_API
    base = {
        "start_date": (start - timedelta(days=1)).isoformat(),
        "end_date": (start + timedelta(days=days)).isoformat(),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    if model:
        base["models"] = model
    if key:
        base["apikey"] = key

    # Probe with one point so an unsupported variable fails fast and cheaply.
    names = _probe(pts[0], names, base, api, log)
    ndays = days + 2
    weight = math.ceil(len(names) / 10) * max(1.0, ndays / 14)
    pace = 0 if key else batch * weight / PER_MINUTE * 60
    log(f"{len(pts)} points x {len(names)} variables, ~{len(pts) * weight:.0f} weighted calls")

    data = []
    for i in range(0, len(pts), batch):
        t0 = time.monotonic()
        chunk = pts[i:i + batch]
        params = dict(base,
                      latitude=",".join(str(p[0]) for p in chunk),
                      longitude=",".join(str(p[1]) for p in chunk),
                      hourly=",".join(names))
        res = _get(params, api, log=log)
        data += res if isinstance(res, list) else [res]
        log(f"  fetched {min(i + batch, len(pts))}/{len(pts)} points")
        if i + batch < len(pts):
            time.sleep(max(0.0, pace - (time.monotonic() - t0)))

    meta = {"model": f"GFS {model or 'seamless'} via Open-Meteo",
            "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "step": step or GRID["step"]}
    run = model_run()
    if run:
        meta["run"] = run
    return {"pts": pts, "data": data, "meta": meta}


def _probe(pt, names, base, api, log):
    names = list(names)
    while True:
        params = dict(base, latitude=pt[0], longitude=pt[1], hourly=",".join(names))
        try:
            _get(params, api, log=log)
            return names
        except _BadVariable as e:
            bad = [n for n in names if re.search(rf"\b{re.escape(n)}\b", str(e))]
            if not bad:
                raise RuntimeError(f"Open-Meteo rejected the request: {e}") from None
            log(f"  Open-Meteo does not serve {bad}; continuing without")
            names = [n for n in names if n not in bad]


def model_run(url="https://api.open-meteo.com/data/ncep_gfs025/static/meta.json"):
    """Best-effort GFS initialisation time from Open-Meteo's model metadata."""
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            ts = json.load(r)["last_run_initialisation_time"]
        return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    except Exception:
        return None
