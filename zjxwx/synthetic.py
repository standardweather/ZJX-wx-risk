"""Synthetic GFS-like grid in the Open-Meteo JSON format, for demos and tests.

The scenario is a late-season cold front sweeping across Georgia and north
Florida with an upper jet streak, post-frontal overrunning cloud (icing
aloft) and diurnal warm-sector thunderstorms over the peninsula, followed
by a quiet spell and a weaker second system. It is not a forecast: images
made from it carry a "SYNTHETIC DATA" banner.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import CAT_LEVELS, GRID, ICING_LEVELS

Z_STD = {925: 760, 850: 1460, 800: 1950, 750: 2470, 700: 3010, 650: 3590, 600: 4210,
         550: 4860, 500: 5570, 450: 6320, 400: 7180, 350: 8120, 300: 9160, 250: 10360,
         200: 11780}
JET_PROFILE = {500: 0.45, 450: 0.55, 400: 0.66, 350: 0.8, 300: 0.93, 250: 1.0, 200: 0.88}
LEVELS = sorted(set(ICING_LEVELS) | set(CAT_LEVELS), reverse=True)


def make_grid(start: date, days: int, step: float = 0.5, seed: int = 7):
    rng = np.random.default_rng(seed)
    lats = np.arange(GRID["lat0"], GRID["lat1"] + 1e-9, step)
    lons = np.arange(GRID["lon0"], GRID["lon1"] + 1e-9, step)
    t0 = datetime.combine(start - timedelta(days=1), datetime.min.time())
    nt = 24 * (days + 2)
    times = [t0 + timedelta(hours=i) for i in range(nt)]
    th = np.arange(nt, dtype=float)[:, None, None] - 24.0   # hours since start 00Z
    LAT = lats[None, :, None]
    LON = lons[None, None, :]

    # Two frontal systems: strong on day 0, weaker on day 3/4.
    sys_t = np.array([6.0, 24 * 3 + 18.0])
    sys_a = np.array([1.0, 0.75])
    dt = th - sys_t[:, None, None, None]
    near = np.argmin(np.abs(dt), axis=0)
    dts = np.take_along_axis(dt, near[None], 0)[0]
    amp = np.sum(sys_a[:, None, None, None] * np.exp(-(dt / 34.0) ** 2), axis=0)
    yf = 33.2 - 1.5 * dts / 24.0                             # front latitude at 82W
    d = LAT - (yf + 0.28 * (LON + 82.0))                     # >0 cold side
    cold = 0.5 * (1 + np.tanh(d / 0.6))

    local_h = (th + 24 * 10 - 4) % 24                       # EDT
    diurnal = np.clip(np.sin(np.pi * (local_h - 11) / 11), 0, None)

    # Surface temperature and freezing level
    t2m = 27.5 - 0.85 * (LAT - 25) - 7.0 * cold * amp - 3 * (1 - diurnal) + 0 * LON
    # Moisture
    band = np.exp(-((d - 0.3) / 0.55) ** 2) * amp
    overrun = np.exp(-((d - 1.6) / 1.1) ** 2) * amp * (d > -0.2)
    warm_moist = (1 - cold) * (LAT < 31.5) * 0.8

    # Convection: warm-sector afternoon cells, plus a line along the front
    cape = (2600 * (1 - cold) * np.clip((31.5 - LAT) / 3, 0, 1) * (0.35 + 0.65 * diurnal)
            * (0.5 + 0.5 * np.clip(amp, 0, 1)) + 600 * band * (1 - cold))
    noise = gaussian_filter(rng.standard_normal((nt, len(lats), len(lons))), (2.5, 1.1, 1.1))
    noise /= noise.std()
    cells = np.clip(noise - 0.25, 0, None) ** 1.2
    conv_pr = 12 * cells * diurnal * np.clip(cape / 2000, 0, 1.2) * (1 - cold) * (LON > -86)
    conv_pr += 9 * band * (1 - cold) * (0.6 + 0.4 * np.sin(1.3 * LON - 0.1 * th) ** 2)
    strat_pr = 2.5 * overrun * np.clip(1.6 - d / 2, 0, 1) * (d > 0)
    precip = np.clip(conv_pr + strat_pr + rng.gamma(0.3, 0.05, conv_pr.shape) * band, 0, None)
    li = 2.0 - cape / 380.0
    cin = -(15 + 90 * (1 - diurnal)) * (1 - cold) - 5

    hourly = {"precipitation": precip, "cape": cape, "lifted_index": li,
              "convective_inhibition": cin, "temperature_2m": t2m}
    units = {"precipitation": "mm", "cape": "J/kg", "lifted_index": "", "convective_inhibition": "J/kg",
             "temperature_2m": "°C", "freezing_level_height": "m"}

    # Jet axis meanders north of the front; speed grows with the system.
    yj = 34.6 + 1.3 * np.sin((LON + 84) / 5.0 - th / 70.0) - 2.4 * np.clip(amp, 0, 1) + 0 * LAT
    jet_core = 22 + 40 * np.clip(amp, 0, 1.2) ** 1.5
    slope = np.gradient(yj, lons, axis=2)
    for p in LEVELS:
        z = Z_STD[p]
        tp = t2m - 6.6 * z / 1000 + 2.5 * (1 - cold) * (z < 1600) - 1.5 * cold * amp * (z > 4000)
        rh = (45 + 35 * warm_moist * np.exp(-z / 2500)
              + 55 * band * np.exp(-z / 9000) + 62 * overrun * (z > 1200) * (z < 6500)
              - 0.002 * z + 6 * np.sin(0.8 * LON + 0.6 * LAT + 0.05 * th))
        rh = np.clip(rh, 5, 100)
        jp = JET_PROFILE.get(p, 0.25 * z / 5570)
        width = 2.4 + 0.4 * np.cos(LON / 3)
        u = 6 + jp * jet_core * np.exp(-((LAT - yj) / width) ** 2) + 6 * jp
        u += -4 * cold * (z < 2000) + 3 * (1 - cold) * (z < 2000)
        v = u * slope * 0.9 + 4 * (1 - cold) * (z < 3500) - 5 * cold * (z < 2500)
        # Heights: thermal wind consistent-ish, higher to the south.
        zz = z + (z / 10000) * (8.0 * (tp - (-6.6 * z / 1000 + 20))) - 2.0 * (LAT - 30) * (z / 1000)
        spd = np.hypot(u, v) * 3.6
        wdir = np.degrees(np.arctan2(-u, -v)) % 360
        hourly[f"temperature_{p}hPa"] = tp
        hourly[f"relative_humidity_{p}hPa"] = rh
        units[f"temperature_{p}hPa"] = "°C"
        units[f"relative_humidity_{p}hPa"] = "%"
        if p in CAT_LEVELS or p == 500:  # 500 hPa kept for the legacy scripts
            hourly[f"wind_speed_{p}hPa"] = spd
            hourly[f"wind_direction_{p}hPa"] = wdir
            hourly[f"geopotential_height_{p}hPa"] = zz
            units[f"wind_speed_{p}hPa"] = "km/h"
            units[f"wind_direction_{p}hPa"] = "°"
            units[f"geopotential_height_{p}hPa"] = "m"
    hourly["freezing_level_height"] = np.clip(t2m / 6.1 * 1000 + 250 * (1 - cold), 0, None)

    tstr = [t.strftime("%Y-%m-%dT%H:%M") for t in times]
    pts, data = [], []
    shape = (nt, len(lats), len(lons))
    arrays = {k: np.round(np.broadcast_to(v, shape), 2) for k, v in hourly.items()}
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            pts.append([round(float(la), 3), round(float(lo), 3)])
            data.append({"latitude": float(la), "longitude": float(lo), "hourly_units": units,
                         "hourly": {"time": tstr, **{k: a[:, i, j].tolist() for k, a in arrays.items()}}})
    meta = {"synthetic": True, "model": "SYNTHETIC GFS-like demo data",
            "run": (t0 + timedelta(hours=18)).strftime("%Y-%m-%dT%H:%MZ"), "step": step}
    return {"pts": pts, "data": data, "meta": meta}


def write(path, start, days, step=0.5):
    gj = make_grid(start, days, step)
    with open(path, "w") as fh:
        json.dump(gj, fh, separators=(",", ":"))
    return path
