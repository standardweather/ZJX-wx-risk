"""Load the Open-Meteo point-grid JSON into dense (time, lat, lon) arrays.

The file format is the one the original scripts used,
``{"pts": [[lat, lon], ...], "data": [<Open-Meteo response>, ...]}``,
optionally with a ``"meta"`` block written by ``zjxwx.fetch``/``synthetic``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# Open-Meteo unit strings -> factor to m/s.
_WIND_TO_MS = {"km/h": 1 / 3.6, "kmh": 1 / 3.6, "m/s": 1.0, "ms": 1.0,
               "kn": 0.514444, "kt": 0.514444, "mph": 0.44704, "mp/h": 0.44704}


@dataclass
class Grid:
    lats: np.ndarray                  # (ny,) ascending
    lons: np.ndarray                  # (nx,) ascending
    times: np.ndarray                 # (nt,) datetime64[m], UTC
    fields: dict = field(repr=False)  # name -> (nt, ny, nx) float, NaN = missing
    units: dict = field(default_factory=dict, repr=False)
    meta: dict = field(default_factory=dict)

    def has(self, name: str) -> bool:
        return name in self.fields

    def get(self, name: str) -> np.ndarray | None:
        return self.fields.get(name)

    def time_index(self, when: datetime) -> int | None:
        t = np.datetime64(when.astimezone(timezone.utc).replace(tzinfo=None), "m")
        hits = np.nonzero(self.times == t)[0]
        return int(hits[0]) if hits.size else None

    def wind(self, level: int) -> tuple[np.ndarray, np.ndarray] | None:
        """(u, v) in m/s at a pressure level; NaN where the model gave nothing."""
        spd = self.get(f"wind_speed_{level}hPa")
        wdir = self.get(f"wind_direction_{level}hPa")
        if spd is None or wdir is None:
            return None
        spd = spd * _WIND_TO_MS.get(self.units.get(f"wind_speed_{level}hPa", "km/h"), 1 / 3.6)
        r = np.radians(wdir)
        return -spd * np.sin(r), -spd * np.cos(r)

    @property
    def model_label(self) -> str:
        return self.meta.get("model", "GFS (Open-Meteo)")

    @property
    def synthetic(self) -> bool:
        return bool(self.meta.get("synthetic"))


def load_grid(path: str) -> Grid:
    with open(path) as fh:
        gj = json.load(fh)
    pts = np.asarray(gj["pts"], dtype=float)
    data = gj["data"]
    if len(pts) != len(data):
        raise ValueError(f"{path}: {len(pts)} points but {len(data)} responses")

    lats = np.unique(pts[:, 0])
    lons = np.unique(pts[:, 1])
    iy = np.searchsorted(lats, pts[:, 0])
    ix = np.searchsorted(lons, pts[:, 1])

    time_strs = data[0]["hourly"]["time"]
    times = np.array(time_strs, dtype="datetime64[m]")
    tpos = {s: i for i, s in enumerate(time_strs)}

    names = [k for k in data[0]["hourly"] if k != "time"]
    fields = {n: np.full((len(times), len(lats), len(lons)), np.nan) for n in names}
    for k, loc in enumerate(data):
        h = loc["hourly"]
        if h["time"] == time_strs:
            sel = slice(None)
            dst = slice(None)
        else:  # tolerate ragged responses by aligning on time strings
            pairs = [(tpos[s], j) for j, s in enumerate(h["time"]) if s in tpos]
            dst, sel = (np.array(p) for p in zip(*pairs)) if pairs else ([], [])
        for n in names:
            vals = h.get(n)
            if vals is None:
                continue
            arr = np.array(vals, dtype=float)  # None -> nan
            fields[n][dst, iy[k], ix[k]] = arr[sel]

    units = dict(data[0].get("hourly_units", {}))
    return Grid(lats=lats, lons=lons, times=times, fields=fields, units=units,
                meta=gj.get("meta", {}))
