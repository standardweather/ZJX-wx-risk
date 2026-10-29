"""Shift windows, hazard summaries and the workload rating."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from . import diagnostics as dx
from .config import CAT_LEVELS, CATEGORY, ICING_LEVELS, JET_LEVEL, LEVEL_FL, TIMEZONE, Shift


# --------------------------------------------------------------------------
# Time windows
# --------------------------------------------------------------------------
@dataclass
class Window:
    shift: Shift
    day: date
    start: datetime      # aware, UTC
    end: datetime        # aware, UTC
    tz: ZoneInfo

    @classmethod
    def for_shift(cls, shift: Shift, day: date, tz: str = TIMEZONE):
        z = ZoneInfo(tz)
        s = datetime.combine(day, time.fromisoformat(shift.start), z)
        e = datetime.combine(day, time.fromisoformat(shift.end), z)
        if e <= s:
            e += timedelta(days=1)
        return cls(shift, day, s.astimezone(timezone.utc), e.astimezone(timezone.utc), z)

    def accum_hours(self):
        """Valid times T of hourly accumulations (T-1h, T] overlapping the shift."""
        t = _ceil_hour(self.start + timedelta(minutes=1))
        out = []
        while t - timedelta(hours=1) < self.end:
            out.append(t)
            t += timedelta(hours=1)
        return out

    def instant_hours(self):
        """Hourly snapshots spanning the shift (inclusive of both ends)."""
        t = self.start.replace(minute=0, second=0, microsecond=0)
        out = []
        while t <= _ceil_hour(self.end):
            out.append(t)
            t += timedelta(hours=1)
        return out

    def local_label(self):
        s, e = self.start.astimezone(self.tz), self.end.astimezone(self.tz)
        return f"{s:%H%M}–{e:%H%M} {s:%Z}"

    def utc_label(self):
        return f"{self.start:%H%M}–{self.end:%H%M} UTC"


def _ceil_hour(t):
    base = t.replace(minute=0, second=0, microsecond=0)
    return base if base == t else base + timedelta(hours=1)


# --------------------------------------------------------------------------
# Full-period diagnostics (computed once per grid)
# --------------------------------------------------------------------------
class Diagnostics:
    def __init__(self, grid, cfg):
        self.grid, self.cfg = grid, cfg
        step_h = _time_step_hours(grid.times)
        tc = cfg["turbulence"]
        layers = dx.ellrod_layers(grid, CAT_LEVELS, smooth=tc.get("smooth_points", 3))
        if not layers:
            raise ValueError("grid lacks the upper-level wind/height fields for turbulence")
        self.dti, self.ti1, which = dx.ellrod_knox_dti(
            layers, tc["dvt_c"], max(1, round(tc["dvt_hours"] / step_h)), tc["dvt_hours"],
            tc.get("dvt_div_scale", 1e6),
            dx.convective_mask(grid.get("precipitation")) if grid.has("precipitation") else None)
        self.dti_fl = np.array([LEVEL_FL[(L["bot"])] + LEVEL_FL[L["top"]] for L in layers]) / 2
        self.dti_layer = which
        self.vws = np.nanmax(np.stack([L["vws_kt_kft"] for L in layers]), axis=0)

        ic = cfg["icing"]
        self.ice, _, self.ice_base, self.ice_top = dx.icing_potential(grid, ICING_LEVELS, ic)
        self.fzra = dx.freezing_precip(grid, ic)
        self.fzl = dx.freezing_level_fl(grid)
        self.ts = dx.thunderstorm_hours(grid, cfg["convection"])


def _time_step_hours(times):
    if len(times) < 2:
        return 1.0
    return float((times[1] - times[0]) / np.timedelta64(1, "h"))


# --------------------------------------------------------------------------
# Window summary
# --------------------------------------------------------------------------
@dataclass
class Summary:
    window: Window
    hours: list                      # instantaneous valid times (UTC)
    precip_in: np.ndarray
    ts_hours: np.ndarray | None
    cape_max: np.ndarray | None
    dti_max: np.ndarray
    dti_fl: np.ndarray               # FL of the layer holding the max
    ice_hours: np.ndarray
    ice_max: np.ndarray
    ice_base: np.ndarray             # lowest / highest FL with icing potential
    ice_top: np.ndarray
    fzra_hours: np.ndarray | None
    fzl_fl: np.ndarray | None
    jet_time: datetime
    jet_u: np.ndarray | None
    jet_v: np.ndarray | None
    series: dict = field(default_factory=dict)  # name -> per-hour airspace fraction
    missing: list = field(default_factory=list)


def summarize(diag: Diagnostics, window: Window, mask=None) -> Summary:
    g, cfg = diag.grid, diag.cfg
    acc = _indices(g, window.accum_hours())
    inst_times = [t for t in window.instant_hours() if g.time_index(t) is not None]
    inst = _indices(g, inst_times)
    if not inst or not acc:
        raise ValueError(f"grid has no data for {window.day} {window.shift.code} shift "
                         f"({window.utc_label()}); grid covers {g.times[0]} to {g.times[-1]} UTC")
    missing = []
    if len(acc) < len(window.accum_hours()):
        missing.append(f"precip hours {len(acc)}/{len(window.accum_hours())}")

    pr = g.get("precipitation")
    precip_in = np.nansum(pr[acc], axis=0) / 25.4

    ts_hours = cape_max = None
    if diag.ts is not None:
        ts_hours = diag.ts[inst].sum(axis=0).astype(float)
        cape_max = np.nanmax(g.get("cape")[inst], axis=0)
    else:
        missing.append("CAPE")

    dti = diag.dti[inst]
    k = np.argmax(dti, axis=0)
    dti_max = np.take_along_axis(dti, k[None], 0)[0]
    layer = np.take_along_axis(diag.dti_layer[inst], k[None], 0)[0]

    thr = cfg["icing"]["potential_threshold"]
    ice_hours = (diag.ice[inst] >= thr).sum(axis=0).astype(float)
    fzra_hours = diag.fzra[inst].sum(axis=0).astype(float) if diag.fzra is not None else None
    fzl = np.nanmean(diag.fzl[inst], axis=0) if diag.fzl is not None else None

    mid = inst_times[len(inst_times) // 2]
    jw = g.wind(JET_LEVEL)
    ji = g.time_index(mid)
    jet_u, jet_v = (jw[0][ji], jw[1][ji]) if jw else (None, None)

    s = Summary(window, inst_times, precip_in, ts_hours, cape_max, dti_max,
                diag.dti_fl[layer], ice_hours, diag.ice[inst].max(axis=0),
                np.fmin.reduce(diag.ice_base[inst], axis=0), np.fmax.reduce(diag.ice_top[inst], axis=0),
                fzra_hours, fzl,
                mid, jet_u, jet_v, missing=missing)
    if mask is not None:
        lvl = cfg["turbulence"]["levels"][0]
        s.series = {
            "Thunderstorms": mask.frac_series(diag.ts[inst], 0.5) if diag.ts is not None else None,
            f"Turbulence (DTI ≥ {lvl})": mask.frac_series(dti, lvl),
            "Icing": mask.frac_series(diag.ice[inst], thr),
        }
    return s


def _indices(g, times):
    idx = [g.time_index(t) for t in times]
    return [i for i in idx if i is not None]


# --------------------------------------------------------------------------
# Workload
# --------------------------------------------------------------------------
@dataclass
class HazardRating:
    name: str
    level: int
    detail: str        # key numbers, always shown
    driver: str = ""   # which rule fired

    @property
    def label(self):
        return CATEGORY[self.level]


@dataclass
class Workload:
    level: int
    hazards: list
    reason: str

    @property
    def label(self):
        return CATEGORY[self.level]


def _rate(field, rules, mask, fmt):
    """Highest category whose any [threshold, area_fraction] rule is met."""
    for level, key in ((2, "high"), (1, "moderate")):
        for thr, frac in rules.get(key, []):
            f = mask.frac_ge(field, thr)
            if f >= frac:
                return level, fmt(thr, f)
    return 0, ""


def assess(s: Summary, mask, cfg) -> Workload:
    wl = cfg["workload"]
    out = []

    if s.ts_hours is not None:
        lvl, why = _rate(s.ts_hours, wl["convection"], mask,
                         lambda t, f: f"TS ≥{t:g} h over {f:.0%} of airspace")
        detail = (f"{mask.frac_ge(s.ts_hours, 1):.0%} of airspace ≥1 TS hour · "
                  f"max {mask.max(s.ts_hours):.0f} h · CAPE max {mask.max(s.cape_max):,.0f} J/kg")
        out.append(HazardRating("Thunderstorms", lvl, detail, why))

    lvl, why = _rate(s.dti_max, wl["turbulence"], mask,
                     lambda t, f: f"DTI ≥{t:g} over {f:.0%} of airspace")
    imax = np.nanargmax(np.where(np.isfinite(s.dti_max), s.dti_max, -1))
    detail = (f"DTI max {mask.max(s.dti_max):.0f} near FL{s.dti_fl.ravel()[imax]:.0f} · "
              f"{mask.frac_ge(s.dti_max, 4):.0%} of airspace ≥4")
    out.append(HazardRating("Turbulence (CAT)", lvl, detail, why))

    ic = wl["icing"]
    lvl, why = _rate(s.ice_hours, ic, mask, lambda t, f: f"icing ≥{t:g} h over {f:.0%} of airspace")
    if s.fzra_hours is not None:
        fz = mask.frac_ge(s.fzra_hours, 1)
        if fz >= ic["fzra_high_frac"]:
            lvl, why = 2, f"freezing precipitation over {fz:.0%} of airspace"
    detail = f"{mask.frac_ge(s.ice_hours, 3):.0%} of airspace ≥3 icing h · max {mask.max(s.ice_hours):.0f} h"
    base, top = mask.sample(s.ice_base), mask.sample(s.ice_top)
    if np.isfinite(base).any():
        detail += f" · FL{np.nanpercentile(base, 10):03.0f}–FL{np.nanpercentile(top, 90):03.0f}"
    if s.fzl_fl is not None:
        detail += f" · FZL ~FL{np.nanmedian(mask.sample(s.fzl_fl)):03.0f}"
    out.append(HazardRating("Icing", lvl, detail, why))

    mm = s.precip_in * 25.4
    lvl, why = _rate(mm, wl["precip"], mask,
                     lambda t, f: f">{t / 25.4:.1f} in over {f:.0%} of airspace")
    detail = f"max {mask.max(s.precip_in):.2f} in · {mask.frac_ge(mm, 12.7):.0%} of airspace >0.5 in"
    out.append(HazardRating("Heavy precipitation", lvl, detail, why))

    level = max(h.level for h in out)
    top = [h for h in out if h.level == level and h.level > 0]
    reason = "; ".join(f"{h.name}: {h.driver}" for h in top) if top else "No hazard reaches moderate thresholds"
    n_mod = sum(h.level == 1 for h in out)
    if level == 1 and n_mod >= wl["compound_moderates"]:
        level = 2
        reason = f"{n_mod} moderate hazards combined (" + ", ".join(h.name for h in out if h.level == 1) + ")"
    return Workload(level, out, reason)

