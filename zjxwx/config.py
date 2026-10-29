"""Static configuration: domain, shifts, levels, hazard thresholds, styling.

Everything a forecaster might want to tune lives here, so the science
modules stay free of magic numbers. Values can be overridden at run time
with a JSON file passed to ``--config`` (see ``load_config``).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Map extent (lon0, lon1, lat0, lat1) and the Open-Meteo sampling grid.
EXTENT = (-88.4, -76.6, 24.3, 35.3)
GRID = {"lat0": 23.5, "lat1": 36.0, "lon0": -89.5, "lon1": -75.0, "step": 0.5}

TIMEZONE = "America/New_York"


@dataclass(frozen=True)
class Shift:
    code: str
    name: str
    start: str  # local HH:MM
    end: str    # local HH:MM (may be past midnight)


SHIFTS = {
    "F": Shift("F", "Day", "05:15", "13:15"),
    "N": Shift("N", "Evening", "13:00", "21:00"),
}

# Pressure levels. Cruise (FL240-FL450) layers for CAT; low/mid levels for icing.
CAT_LEVELS = (400, 350, 300, 250, 200)          # layers taken between neighbours
# 925-450 hPa (~FL025-FL210). Kept to 10 levels so a fetch is 40 variables,
# i.e. 4 Open-Meteo weighted calls per point (see fetch.py).
ICING_LEVELS = (925, 850, 800, 750, 700, 650, 600, 550, 500, 450)
JET_LEVEL = 250

# Approximate standard-atmosphere flight levels, for labelling only.
LEVEL_FL = {925: 25, 850: 50, 800: 65, 750: 80, 700: 100, 650: 115, 600: 140,
            550: 160, 500: 180, 450: 210, 400: 240, 350: 270, 300: 300,
            250: 340, 200: 390}

DEFAULTS = {
    "turbulence": {
        # Ellrod & Knapp (1992) TI1 units: 1e-7 s^-2. AWC-style guidance bins.
        "levels": [4, 8, 12],
        # Ellrod & Knox (2010) divergence-trend term: DVT = C * dDIV over dt.
        # C = 0.1 was "subjectively assigned" for a 6 h interval; the authors
        # proposed scaling C with the interval, which we do (C * dt/6h).
        "dvt_c": 0.1,
        "dvt_hours": 6,
        "dvt_div_scale": 1e6,   # divergence in 1e-6 s^-1 (see diagnostics.ellrod_knox_dti)
        "smooth_points": 3,     # 3x3 box smoothing of TI and divergence
    },
    "icing": {
        # Temperature interest (deg C -> 0..1), loosely after CIP/FIP
        # (Bernstein et al. 2005): supercooled liquid is most frequent
        # between about -3 and -12 C and rarer below -20 C.
        "t_interest": [[1.0, 0.0], [0.0, 0.0], [-3.0, 1.0], [-12.0, 1.0],
                       [-20.0, 0.3], [-25.0, 0.0]],
        # Moisture interest from RH (w.r.t. liquid, as GFS reports it).
        "rh_interest": [[70.0, 0.0], [85.0, 0.6], [95.0, 1.0]],
        "potential_threshold": 0.5,     # "icing hour" if column max >= this
        # Freezing precip: sfc T <= 0 C, precip >= 0.25 mm/h, warm nose > +1 C.
        "fzra_t2m": 0.0, "fzra_precip": 0.25, "fzra_warm_nose": 1.0,
    },
    "convection": {
        # A thunderstorm hour needs instability, a lifting signal and no strong cap.
        "cape_min": 500.0,       # J/kg
        "li_max": -2.0,          # K
        "cin_min": -100.0,       # J/kg (more negative = stronger cap)
        "precip_min": 1.0,       # mm/h (convective or total)
        "cape_strong": 2000.0,
    },
    # Workload rules use the fraction of ZJX airspace (area-weighted) meeting a
    # condition. Each hazard maps to 0 LOW / 1 MODERATE / 2 HIGH.
    "workload": {
        "convection": {"high": [[3, 0.20], [1, 0.40]], "moderate": [[1, 0.08]]},
        "turbulence": {"high": [[8, 0.15], [12, 0.03]], "moderate": [[4, 0.10], [8, 0.03]]},
        "icing": {"high": [[6, 0.40]], "moderate": [[3, 0.25]], "fzra_high_frac": 0.02},
        "precip": {"high": [[25.4, 0.10]], "moderate": [[12.7, 0.10]]},
        # Two or more MODERATE hazards together raise the overall rating to HIGH.
        "compound_moderates": 2,
    },
}

CATEGORY = ["LOW", "MODERATE", "HIGH"]

# Palette: hazard hues follow AWC conventions (convection red, turbulence
# orange/brown, icing blue), precipitation in WPC-style greens.
STYLE = {
    "navy": "#0b2545",
    "ink": "#1d2433",
    "muted": "#5b6577",
    "rule": "#d5dae3",
    "paper": "#ffffff",
    "ocean": "#e8eef4",
    "land": "#f7f6f2",
    "coast": "#7d8796",
    "state": "#a9b1bd",
    "boundary": "#14213d",
    "cat": {"LOW": "#2e7d4f", "MODERATE": "#d98c00", "HIGH": "#c62828"},
    "precip_colors": ["#e3f4dc", "#bfe5b0", "#8ccf7e", "#4fae5b", "#1f8a45",
                      "#0b6a3a", "#2c5e8f", "#5a3e9b"],
    "precip_levels_in": [0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 1.50, 2.00, 3.00],
    "ts": "#d7263d",
    "turb": ["#f6c87a", "#e8871e", "#a6400b"],
    "ice": "#1565c0",
    "jet": "#6b5b95",
    "fzlvl": "#00838f",
}

AIRPORTS = {  # id: (lon, lat)
    "JAX": (-81.688, 30.494), "SAV": (-81.202, 32.127), "CHS": (-80.040, 32.899),
    "TLH": (-84.350, 30.397), "MCO": (-81.309, 28.429), "TPA": (-82.533, 27.976),
    "DAB": (-81.058, 29.180), "PNS": (-87.187, 30.473), "MIA": (-80.291, 25.796),
    "ATL": (-84.428, 33.637),
}

DISCLAIMER = ("Unofficial planning aid generated automatically from raw GFS model output. "
              "Not a substitute for official weather information: consult AWC G-AIRMETs, "
              "SIGMETs, CWAs and the ZJX CWSU. ZJX boundary from VATSIM vatspy-data-project "
              "(approximate, not FAA-official).")


def load_config(path: str | None) -> dict:
    """Return DEFAULTS deep-merged with an optional user JSON file."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        _merge(cfg, json.loads(Path(path).read_text()))
    return cfg


def _merge(base: dict, over: dict) -> None:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
