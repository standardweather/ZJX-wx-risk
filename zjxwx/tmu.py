"""OM/TMU/OS planning maps: one simple ZJX map per period with HAZARD and
AWARENESS areas for the Traffic Management Unit.

HAZARD areas (solid):      thunderstorms, severe clear-air turbulence,
                           severe icing.
AWARENESS areas (hatched): IFR/LIFR, moderate turbulence, moderate icing,
                           low-level wind shear.

Sources. NBM (``nbm.py``) is used wherever it has the field: thunder
probability, IFR/LIFR probabilities, LLWS and freezing rain. NBM has no
in-flight icing aloft and no usable upper-level turbulence, so those come
from GFS through the existing diagnostics (Ellrod-Knox DTI FL240-FL390,
CIP-style icing potential FL025-FL210). Without an NBM file the maps still
render: thunderstorms fall back to the GFS thunderstorm signal, freezing
rain to the GFS warm-nose test, and IFR/LIFR and LLWS are marked
unavailable.

Every area means "the condition is met in at least one hour of the
period"; the side panel gives the hours, the share of ZJX and altitudes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.ndimage import uniform_filter
from shapely.geometry import Point

from . import diagnostics as dx
from .analysis import Window, summarize
from .config import AIRPORTS, ICING_LEVELS, STYLE as S, TIMEZONE, TMU_PERIODS
from .geo import _polygon_path
from .render import HALO, PC, _fine, _header, _map_axes, _overlay_geography, _tint

C = S["tmu"]


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Element:
    key: str
    kind: str              # "HAZARD" or "AWARENESS"
    name: str
    color: str
    hatch: str | None
    source: str            # "NBM" or "GFS"
    available: bool = True
    frac: float = 0.0      # share of ZJX airspace affected at some time in the period
    hours: list = field(default_factory=list)  # (start, end) UTC of each affected hour
    detail: str = ""
    plot: tuple | None = None  # (X, Y, Z, level) for contouring, Z >= level is "in"

    @property
    def present(self):
        return self.available and self.frac > 0


class _NbmMask:
    """Area weights of the NBM points inside the ZJX boundary."""

    def __init__(self, nbm, boundary):
        inside = _polygon_path(boundary).contains_points(
            np.c_[nbm.lons.ravel(), nbm.lats.ravel()]).reshape(nbm.lats.shape)
        w = np.where(inside, np.cos(np.radians(nbm.lats)), 0.0)
        self.w = w / w.sum()

    def frac(self, ind):
        return float(np.sum(self.w * np.nan_to_num(ind, nan=0.0)))

    def series(self, cube):
        return np.array([self.frac(c) for c in cube])


def _smooth_ind(ind):
    """Indicator -> 3x3-averaged float, so a contour at 0.5 drops single-pixel specks."""
    return uniform_filter(ind.astype(float), size=3, mode="nearest")


def _span(hours, tz):
    """'0600–1100L' (or several spans) from a list of (start, end) UTC hours."""
    if not hours:
        return ""
    hours = sorted(hours)
    spans, (a, b) = [], hours[0]
    for s, e in hours[1:]:
        if s - b <= timedelta(hours=1):
            b = max(b, e)
        else:
            spans.append((a, b))
            a, b = s, e
    spans.append((a, b))
    return ", ".join(f"{x.astimezone(tz):%H%M}L" if x == y else
                     f"{x.astimezone(tz):%H%M}–{y.astimezone(tz):%H%M}L" for x, y in spans[:3])


def _hours_from_series(series, times, min_frac, accum):
    out = []
    for f, t in zip(series, times):
        if f >= min_frac:
            # an accumulation valid at t covers the hour before it; a snapshot is the instant t
            out.append((t - timedelta(hours=1), t) if accum else (t, t))
    return out


# --------------------------------------------------------------------------
# Analysis for one period
# --------------------------------------------------------------------------
def _severe_ice_cube(diag):
    """Hours with a deep saturated supercooled layer (cached on diag)."""
    if not hasattr(diag, "_tmu_ice_sev"):
        t = diag.cfg["tmu"]
        _, per, _, _ = dx.icing_potential(diag.grid, ICING_LEVELS, diag.cfg["icing"])
        n = sum((p >= t["ice_severe_potential"]).astype(int) for p in per.values())
        diag._tmu_ice_sev = n >= t["ice_severe_levels"]
    return diag._tmu_ice_sev


def analyse(window, diag, mask, nbm=None, nbm_mask=None, cfg=None, hrrr=None):
    """Return the list of Elements (hazards first) for one Window."""
    cfg = cfg or diag.cfg
    t = cfg["tmu"]
    grid, tz, minf = diag.grid, window.tz, t["min_area"]
    s = summarize(diag, window, mask)
    inst = [grid.time_index(h) for h in s.hours]
    inst_times = s.hours

    def gfs_elem(el, cube, thr, period_field, detail):
        """Fill an Element from a GFS (nt, ny, nx) cube over the window's instants."""
        X, Y, Z = _fine(grid, period_field)
        el.plot = (X, Y, Z, thr)
        el.frac = mask.frac_ge(period_field, thr)
        el.hours = _hours_from_series(mask.frac_series(cube, thr), inst_times, minf, accum=False)
        if el.frac < minf:
            el.frac, el.hours = 0.0, []
        el.detail = detail() if el.frac else ""
        return el

    # NBM time indices for the window.
    nb_inst = nb_acc = None
    if nbm is not None:
        nb_inst = [(nbm.index(h), h) for h in window.instant_hours()]
        nb_inst = [(i, h) for i, h in nb_inst if i is not None]
        nb_acc = [(nbm.index(h), h) for h in window.accum_hours()]
        nb_acc = [(i, h) for i, h in nb_acc if i is not None]
        if not nb_inst or not nb_acc:
            nbm = None

    def nbm_elem(el, cube, times, accum, detail):
        """Fill an Element from an NBM boolean cube (nt, ny, nx)."""
        any_ = cube.any(axis=0)
        el.plot = (nbm.lons, nbm.lats, _smooth_ind(any_), 0.5)
        el.frac = nbm_mask.frac(any_)
        el.hours = _hours_from_series(nbm_mask.series(cube), times, minf, accum)
        if el.frac < minf:
            el.frac, el.hours = 0.0, []
        el.detail = detail() if el.frac else ""
        return el

    out = []

    # ---- HAZARD: thunderstorms -------------------------------------------
    out.append(_thunder(window, diag, s, inst, mask, nbm, hrrr, t, minf))

    # ---- HAZARD: severe clear-air turbulence -----------------------------
    sev, mod = t["turb_severe_dti"], t["turb_moderate_dti"]

    def turb_detail(thr):
        def f():
            fl = mask.sample(np.where(s.dti_max >= thr, s.dti_fl, np.nan))
            fl = fl[np.isfinite(fl)]
            lo, hi = (int(fl.min() - 15), int(fl.max() + 15)) if fl.size else (240, 390)
            return f"DTI ≥ {thr:g} (max {mask.max(s.dti_max):.0f}) · ~FL{max(lo, 240):03d}–FL{min(hi, 390):03d}"
        return f

    el = Element("turb_sev", "HAZARD", "Severe turbulence (CAT)", C["turb_sev"], None, "GFS")
    out.append(gfs_elem(el, diag.dti[inst], sev, s.dti_max, turb_detail(sev)))
    ext = t["turb_extreme_dti"]
    el = Element("turb_ext", "HAZARD", "Extreme turbulence (CAT)", C["turb_ext"], None, "GFS")
    out.append(gfs_elem(el, diag.dti[inst], ext, s.dti_max, turb_detail(ext)))

    # ---- HAZARD: severe icing (freezing rain or deep supercooled layer) --
    sev_cube = _severe_ice_cube(diag)[inst].astype(float)
    sev_hours = sev_cube.sum(axis=0)
    el = Element("ice_sev", "HAZARD", "Severe icing", C["ice_sev"], None, "GFS")
    if nbm is not None:
        idx = [i for i, _ in nb_acc]
        fz = np.nan_to_num(nbm.get("fzra")[idx]) >= t["fzra_mm"]
        fz_any = fz.any(axis=0)
        fz_frac = nbm_mask.frac(fz_any)
        fz_hours = _hours_from_series(nbm_mask.series(fz), [h for _, h in nb_acc], minf, True)
        el.source = "NBM + GFS"
    else:
        fz_any = None
        fz_frac, fz_hours = 0.0, []
        if s.fzra_hours is not None:
            fz_frac = mask.frac_ge(s.fzra_hours, 0.5)
            fz_hours = _hours_from_series(mask.frac_series(diag.fzra[inst].astype(float), 0.5),
                                          inst_times, minf, False)
    deep_frac = mask.frac_ge(sev_hours, 0.5)
    deep_hours = _hours_from_series(mask.frac_series(sev_cube, 0.5), inst_times, minf, False)
    X, Y, Z = _fine(grid, sev_hours)
    el.plot = (X, Y, Z, 0.5)
    el.extra_plots = []
    if fz_any is not None:
        el.extra_plots.append((nbm.lons, nbm.lats, _smooth_ind(fz_any), 0.5))
    elif s.fzra_hours is not None:
        el.extra_plots.append((*_fine(grid, s.fzra_hours), 0.5))
    bits = []
    if fz_frac >= minf:
        bits.append("freezing rain")
    if deep_frac >= minf:
        b, tp = mask.sample(np.where(sev_hours > 0, s.ice_base, np.nan)), \
            mask.sample(np.where(sev_hours > 0, s.ice_top, np.nan))
        rng = (f" FL{np.nanmin(b):03.0f}–FL{np.nanmax(tp):03.0f}"
               if np.isfinite(b).any() and np.isfinite(tp).any() else "")
        bits.append(f"deep supercooled layer{rng}")
    el.frac = min(1.0, (fz_frac if fz_frac >= minf else 0) + (deep_frac if deep_frac >= minf else 0))
    el.hours = sorted((fz_hours if fz_frac >= minf else []) + (deep_hours if deep_frac >= minf else []))
    el.detail = " · ".join(bits) if bits else ""
    el.detail = el.detail[:1].upper() + el.detail[1:]
    out.append(el)

    # ---- AWARENESS: IFR / LIFR -------------------------------------------
    el = Element("ifr", "AWARENESS", "IFR / LIFR", C["ifr"], "////", "NBM")
    if nbm is None:
        el.available = False
        out.append(el)
    else:
        idx = [i for i, _ in nb_inst]
        ifr = np.fmax(nbm.get("ceil_lt1000")[idx], nbm.get("vis_lt3")[idx])
        lifr = np.fmax(nbm.get("ceil_lt500")[idx], nbm.get("vis_lt1")[idx])
        ifr_c = np.nan_to_num(ifr) >= t["ifr_prob"]
        lifr_c = np.nan_to_num(lifr) >= t["lifr_prob"]
        lifr_frac = nbm_mask.frac(lifr_c.any(axis=0))
        nbm_elem(el, ifr_c, [h for _, h in nb_inst], False,
                 lambda: f"P(IFR) ≥ {t['ifr_prob']}%" + (f" · LIFR over {lifr_frac:.0%} of ZJX"
                                                       if lifr_frac >= minf else "") + " · land only")
        el.lifr_plot = (nbm.lons, nbm.lats, _smooth_ind(lifr_c.any(axis=0)), 0.5) if lifr_frac >= minf else None
        el.threshold = t["ifr_prob"]
        el.airports = _airport_ifr(nbm, ifr, lifr, [h for _, h in nb_inst], t, tz, mask.boundary)
        out.append(el)

    # ---- AWARENESS: moderate turbulence ----------------------------------
    el = Element("turb_mod", "AWARENESS", "Moderate turbulence (CAT)", C["turb_mod"], "\\\\\\\\", "GFS")
    out.append(gfs_elem(el, diag.dti[inst], mod, s.dti_max, turb_detail(mod)))

    # ---- AWARENESS: moderate icing ---------------------------------------
    thr = cfg["icing"]["potential_threshold"]
    nh = t["ice_moderate_hours"]

    def ice_detail():
        inn = s.ice_hours >= nh
        b = mask.sample(np.where(inn, s.ice_base, np.nan))
        tp = mask.sample(np.where(inn, s.ice_top, np.nan))
        txt = f"Icing potential ≥ {thr:g} for ≥ {nh} h"
        if np.isfinite(b).any():
            txt += f" · FL{np.nanpercentile(b, 10):03.0f}–FL{np.nanpercentile(tp, 90):03.0f}"
        if s.fzl_fl is not None:
            txt += f" · FZL ~FL{np.nanmedian(mask.sample(s.fzl_fl)):03.0f}"
        return txt

    el = Element("ice_mod", "AWARENESS", "Moderate icing", C["ice_mod"], "////", "GFS")
    cube = (diag.ice[inst] >= thr).astype(float)
    out.append(_ice_mod(el, s, cube, mask, nh, inst_times, minf, grid, ice_detail))

    # ---- AWARENESS: low-level wind shear ---------------------------------
    el = Element("llws", "AWARENESS", "Low-level wind shear", C["llws"], "....", "NBM")
    if nbm is None:
        el.available = False
        out.append(el)
    else:
        idx = [i for i, _ in nb_inst]
        hgt = np.nan_to_num(nbm.get("llws_hgt")[idx])
        spd = np.nan_to_num(nbm.get("llws_spd")[idx])
        cube = hgt > 0
        hrs = cube.sum(axis=0)
        el.plot = (nbm.lons, nbm.lats, _smooth_ind(hrs >= t["llws_hours"]), 0.5)
        el.frac = nbm_mask.frac(hrs >= t["llws_hours"])
        el.hours = _hours_from_series(nbm_mask.series(cube), [h for _, h in nb_inst], minf, False)
        if el.frac < minf:
            el.frac, el.hours = 0.0, []
        else:
            inz = (nbm_mask.w[None] > 0) & cube
            kt = np.nanmax(np.where(inz, spd, np.nan)) * 1.943844
            el.detail = f"NBM LLWS below 2,000 ft AGL · wind at shear height up to {kt:.0f} kt"
        out.append(el)
    return s, out


class _Nearest:
    """Nearest-neighbour lookup from a curvilinear (lat, lon) grid to target points."""

    def __init__(self, lats, lons, qlat, qlon, max_deg=0.1):
        from scipy.spatial import cKDTree

        k = np.cos(np.radians(30.0))
        tree = cKDTree(np.c_[np.ravel(lats), np.ravel(lons) * k])
        d, self.idx = tree.query(np.c_[np.ravel(qlat), np.ravel(qlon) * k])
        self.ok = d <= max_deg
        self.shape = np.shape(qlat)

    def __call__(self, field):
        v = np.ravel(field)[self.idx] & self.ok
        return v.reshape(self.shape)


def _display_grid(res=0.05):
    from .config import EXTENT

    x0, x1, y0, y1 = EXTENT
    return np.meshgrid(np.arange(x0 - 1, x1 + 1, res), np.arange(y0 - 1, y1 + 1, res))


_CACHE = {}


def _nearest(src, key, qlat, qlon):
    k = (id(src), key)
    if k not in _CACHE:
        _CACHE[k] = _Nearest(src.lats, src.lons, qlat, qlon)
    return _CACHE[k]


def _thunder(window, diag, s, inst, mask, nbm, hrrr, t, minf):
    """Thunderstorm hazard: union of HRRR, NBM and GFS signals.

    HRRR (convection-allowing) gives where storms are forecast: echo top
    >= hrrr_top_kft or any lightning threat, widened by hrrr_radius_km for
    placement error. NBM adds its calibrated thunder probability, and GFS
    its CAPE + LI + precipitation signal. Any one of them is enough: for
    traffic management a missed line of 45 kft tops costs far more than an
    over-drawn area, and on 6 Nov 2026 NBM alone missed exactly that.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import maximum_filter

    grid = diag.grid
    DX, DY = _display_grid()
    plat, plon = mask.pts[:, 0], mask.pts[:, 1]
    hours = s.hours
    disp = np.zeros(DX.shape, bool)
    hourly = np.zeros((len(hours), len(plat)), bool)
    used, bits = [], []

    def add(name, cubes, detail):
        """cubes: list of (hour index, disp bool, pts bool)."""
        hit = False
        for k, d, p in cubes:
            disp[:] |= d
            hourly[k] |= p
            hit |= bool(np.sum(mask.w * p) >= minf)
        used.append(name)
        if hit:
            bits.append(detail())

    if hrrr is not None:
        top = hrrr.get("retop") * 3.28084 / 1000.0
        lt = hrrr.get("ltng")
        ind = (np.nan_to_num(top) >= t["hrrr_top_kft"]) | (np.nan_to_num(lt) > t["hrrr_ltng"])
        size = 2 * int(round(t["hrrr_radius_km"] / hrrr.meta.get("stride_km", 6))) + 1
        nd = _nearest(hrrr, "disp", DY, DX)
        npt = _nearest(hrrr, "pts", plat, plon)
        cubes, tops = [], []
        for k, h in enumerate(hours):
            i = hrrr.index(h)
            if i is None:
                continue
            a = maximum_filter(ind[i], size=size)
            cubes.append((k, nd(a), npt(a)))
            tops.append(np.where(npt(ind[i]), np.ravel(top[i])[npt.idx], np.nan))
        tmax = np.nanmax(tops) if tops and np.isfinite(tops).any() else np.nan
        add("HRRR", cubes, lambda: "HRRR storms" + (f", tops to {tmax:.0f} kft" if np.isfinite(tmax) else ""))
    if nbm is not None:
        p = nbm.get("tstm")
        nd = _nearest(nbm, "disp", DY, DX)
        npt = _nearest(nbm, "pts", plat, plon)
        cubes, peak = [], 0.0
        for k, h in enumerate(hours):
            i = nbm.index(h)
            if i is None or h <= window.start:
                continue  # an accumulation valid at the start covers the hour before
            a = np.nan_to_num(p[i]) >= t["ts_prob"]
            cubes.append((k, nd(a), npt(a)))
            peak = max(peak, float(np.nanmax(np.where(npt.ok, np.ravel(p[i])[npt.idx], np.nan))))
        add("NBM", cubes, lambda: f"NBM P(thunder) to {peak:.0f}%")
    if diag.ts is not None:
        cubes = []
        for k, gi in enumerate(inst):
            f = RegularGridInterpolator((grid.lats, grid.lons), diag.ts[gi].astype(float),
                                        bounds_error=False, fill_value=0.0)
            cubes.append((k, f((DY, DX)) >= 0.5, f(mask.pts) >= 0.5))
        add("GFS", cubes, lambda: f"GFS signal, CAPE to {mask.max(s.cape_max):,.0f} J/kg")

    el = Element("ts", "HAZARD", "Thunderstorms", C["ts"], None, " + ".join(used) or "none")
    if not used:
        el.available = False
        return el
    el.plot = (DX, DY, _smooth_ind(disp), 0.5)
    el.frac = float(np.sum(mask.w * hourly.any(axis=0)))
    series = hourly.astype(float) @ mask.w
    el.hours = _hours_from_series(series, hours, minf, accum=False)
    if el.frac < minf:
        el.frac, el.hours = 0.0, []
    el.detail = " · ".join(bits) if el.frac else ""
    return el


def _ice_mod(el, s, cube, mask, nh, times, minf, grid, detail):
    X, Y, Z = _fine(grid, s.ice_hours)
    el.plot = (X, Y, Z, nh - 0.5)
    el.frac = mask.frac_ge(s.ice_hours, nh - 0.5)
    el.hours = _hours_from_series(mask.frac_series(cube, 0.5), times, minf, False)
    if el.frac < minf:
        el.frac, el.hours = 0.0, []
    el.detail = detail() if el.frac else ""
    return el


def _airport_ifr(nbm, ifr, lifr, times, t, tz, boundary):
    """Per-airport IFR/LIFR peak probability and hours (nearest NBM point)."""
    rows = []
    for name, (x, y) in AIRPORTS.items():
        if not boundary.contains(Point(x, y)):
            continue
        d = (nbm.lats - y) ** 2 + ((nbm.lons - x) * np.cos(np.radians(y))) ** 2
        j = np.unravel_index(np.argmin(d), d.shape)
        pi, pl = ifr[(slice(None),) + j], lifr[(slice(None),) + j]
        if not np.isfinite(pi).any():
            continue
        hrs = [(h, h) for h, v in zip(times, pi) if v >= t["ifr_prob"]]
        if not hrs:
            continue
        cat = "LIFR" if np.nanmax(pl) >= t["lifr_prob"] else "IFR"
        rows.append((name, cat, float(np.nanmax(pi)), float(np.nanmax(pl)), _span(hrs, tz)))
    return rows


# --------------------------------------------------------------------------
# Generalising areas for display
# --------------------------------------------------------------------------
_DISP = {}


def _disp_grid():
    """Regular ~5 km lon/lat grid over the map, shared by every element."""
    if "g" not in _DISP:
        DX, DY = _display_grid(0.05)
        _DISP["g"] = (DX, DY)
    return _DISP["g"]


def _to_disp(plot):
    """Boolean 'in area' on the display grid from any element plot tuple."""
    X, Y, Z, lvl = plot
    DX, DY = _disp_grid()
    key = (X.shape, float(np.nanmin(X)), float(np.nanmin(Y)))
    if key not in _CACHE:
        _CACHE[key] = _Nearest(Y, X, DY, DX, max_deg=0.2)
    return _CACHE[key](np.nan_to_num(Z, nan=-1) >= lvl)


def _generalise(ind, g):
    """Grid-point detail -> broad areas: close gaps, drop small blobs, smooth edges.

    Returns a float field to contour at 0.5.
    """
    from scipy.ndimage import binary_closing, gaussian_filter, label

    km = 5.5  # display grid spacing
    r = max(1, int(round(g["close_km"] / km)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    a = binary_closing(ind, structure=(xx ** 2 + yy ** 2) <= r * r, border_value=0)
    min_cells = g["min_area_km2"] / (km * km)
    lab, n = label(a)
    if n:  # drop small areas
        keep = np.bincount(lab.ravel()) >= min_cells
        keep[0] = False
        a = keep[lab]
    lab, n = label(~a)
    if n:  # fill small holes
        small = np.bincount(lab.ravel()) < min_cells
        small[0] = False
        a = a | small[lab]
    return gaussian_filter(a.astype(float), g["smooth_km"] / km)


def generalise(elements, mask, cfg):
    """Replace each element's plot with a smoothed, de-speckled display field
    and recompute its ZJX coverage from what is drawn."""
    g = cfg["tmu"]
    DX, DY = _disp_grid()
    x0, y0 = DX[0, 0], DY[0, 0]
    iy = np.clip(np.round((mask.pts[:, 0] - y0) / 0.05).astype(int), 0, DX.shape[0] - 1)
    ix = np.clip(np.round((mask.pts[:, 1] - x0) / 0.05).astype(int), 0, DX.shape[1] - 1)
    for el in elements:
        if el.plot is None or not el.available:
            continue
        ind = _to_disp(el.plot)
        for extra in getattr(el, "extra_plots", []):
            ind |= _to_disp(extra)
        el.extra_plots = []
        Z = _generalise(ind, g)
        el.plot = (DX, DY, Z, 0.5)
        frac = float(np.sum(mask.w * (Z[iy, ix] >= 0.5)))
        if frac < cfg["tmu"]["min_area"]:
            el.frac, el.hours = 0.0, []
        else:
            el.frac = frac
        lp = getattr(el, "lifr_plot", None)
        if lp is not None:
            Zl = _generalise(_to_disp(lp), g)
            el.lifr_plot = (DX, DY, Zl, 0.5) if np.sum(mask.w * (Zl[iy, ix] >= 0.5)) >= g["min_area"] else None
        el.short = _short(el)


def _short(el):
    """A few words for the panel: altitudes or tops, nothing else."""
    import re

    d = el.detail or ""
    if el.key == "ts":
        m = re.search(r"tops to (\d+) kft", d)
        return f"Tops to {m.group(1)} kft" if m else ""
    if el.key == "llws":
        return "Below 2,000 ft AGL"
    if el.key == "ifr":
        return "LIFR at times" if getattr(el, "lifr_plot", None) is not None else ""
    if el.key == "ice_sev" and "freezing rain" in d.lower():
        return "Freezing rain"
    m = re.search(r"FL(\d{3})–FL(\d{3})", d)
    if not m:
        return ""
    lo, hi = (int(round(int(v) / 10.0)) * 10 for v in m.groups())
    return f"FL{lo:03d}–{hi:03d}"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _draw(ax, el, z):
    if el.plot is None:
        return
    X, Y, Z, lvl = el.plot
    if not el.present or not np.isfinite(Z).any() or np.nanmax(Z) < lvl:
        return
    Z = np.nan_to_num(Z, nan=-1)
    tint = _tint(el.color, 0.30 if el.kind == "AWARENESS" else 0.55)
    if el.kind == "HAZARD":
        ax.contourf(X, Y, Z, levels=[lvl, 1e9], colors=[el.color], alpha=0.42, transform=PC, zorder=z)
        ax.contour(X, Y, Z, levels=[lvl], colors=[el.color], linewidths=1.8, transform=PC, zorder=z + 0.05)
        for X2, Y2, Z2, l2 in getattr(el, "extra_plots", []):
            if np.nanmax(Z2) >= l2:
                Z2 = np.nan_to_num(Z2, nan=-1)
                ax.contourf(X2, Y2, Z2, levels=[l2, 1e9], colors=[el.color], alpha=0.42, transform=PC, zorder=z)
                ax.contour(X2, Y2, Z2, levels=[l2], colors=[el.color], linewidths=1.8, transform=PC,
                           zorder=z + 0.05)
    else:
        ax.contourf(X, Y, Z, levels=[lvl, 1e9], colors=[tint], alpha=0.55, transform=PC, zorder=z)
        h = ax.contourf(X, Y, Z, levels=[lvl, 1e9], colors="none", hatches=[el.hatch], transform=PC,
                        zorder=z + 0.02)
        h.set_edgecolor(el.color)
        h.set_linewidth(0)
        ax.contour(X, Y, Z, levels=[lvl], colors=[el.color], linewidths=1.2, linestyles=[(0, (5, 2.5))],
                   transform=PC, zorder=z + 0.05)
        lp = getattr(el, "lifr_plot", None)
        if lp is not None:
            X2, Y2, Z2, l2 = lp
            Z2 = np.nan_to_num(Z2, nan=-1)
            h = ax.contourf(X2, Y2, Z2, levels=[l2, 1e9], colors="none", hatches=["xxxx"], transform=PC,
                            zorder=z + 0.03)
            h.set_edgecolor(el.color)
            h.set_linewidth(0)
            ax.contour(X2, Y2, Z2, levels=[l2], colors=[el.color], linewidths=1.6, transform=PC,
                       zorder=z + 0.05)


def _swatch(el):
    """Patch style matching how an element is drawn on the map."""
    if not el.present:
        return dict(facecolor="#e9ecf1", edgecolor=S["rule"], lw=0.8)
    if el.kind == "HAZARD":
        return dict(facecolor=_tint(el.color, 0.42), edgecolor=el.color, lw=1.8)
    return dict(facecolor=_tint(el.color, 0.17), edgecolor=el.color, hatch=el.hatch, lw=1.2,
                ls=(0, (5, 2.5)))


def render(window, summary, elements, boundary, out, sources, issued=None, dpi=170, synthetic=False):
    issued = issued or datetime.now(timezone.utc)
    w = window
    fig = plt.figure(figsize=(14, 9.2), facecolor=S["paper"])
    loc0, loc1 = w.start.astimezone(w.tz), w.end.astimezone(w.tz)
    title = f"{w.shift.name}  {loc0:%H%M}–{loc1:%H%M}L"
    day = f"{loc0:%a %d %b %Y}" + (f" – {loc1:%a %d %b}" if loc1.date() != loc0.date() else "")
    y = _header(fig, "ZJX ARTCC  ·  OM / TMU / OS  ·  HAZARD & AWARENESS AREAS",
                f"{title}   ·   {day}",
                f"{w.local_label()}   ·   {w.utc_label()}   ·   "
                f"Issued {issued:%a %d %b %H%MZ}", None, height=0.105)

    ax = _map_axes(fig, [0.015, 0.075, 0.60, y - 0.09])
    order = ["ice_mod", "turb_mod", "llws", "ifr", "ice_sev", "turb_sev", "turb_ext", "ts"]
    by = {e.key: e for e in elements}
    for k, key in enumerate(order):
        if key in by:
            _draw(ax, by[key], 2 + k * 0.2)
    _overlay_geography(ax, boundary)
    if not any(e.present for e in elements):
        ax.text(0.5, 0.06, "No hazard or awareness areas forecast in ZJX", transform=ax.transAxes,
                ha="center", fontsize=12, weight="bold", color=S["cat"]["LOW"], zorder=9,
                path_effects=HALO)
    if synthetic:
        ax.text(0.5, 0.5, "SYNTHETIC DEMO DATA", transform=ax.transAxes, ha="center", va="center",
                fontsize=30, color="#b3261e", alpha=0.25, weight="bold", rotation=30, zorder=10)

    _panel(fig, [0.645, 0.075, 0.335, y - 0.09], elements, w.tz)
    fig.add_artist(Line2D([0.015, 0.98], [0.045, 0.045], transform=fig.transFigure, color=S["rule"], lw=0.8))
    fig.text(0.015, 0.024, "Unofficial Briefing Aid - Automatically Generated", fontsize=9, color=S["muted"],
             va="center")
    fig.savefig(out, dpi=dpi, facecolor=S["paper"])
    plt.close(fig)
    return out


def _panel(fig, rect, elements, tz):
    x, y0, wdt, h = rect
    yy = y0 + h - 0.005
    for kind, head in (("HAZARD", "HAZARD AREAS"), ("AWARENESS", "AWARENESS AREAS")):
        items = [e for e in elements if e.kind == kind]
        n_on = sum(e.present for e in items)
        fig.text(x, yy, head, fontsize=12.5, weight="bold", va="top",
                 color=S["cat"]["HIGH"] if kind == "HAZARD" and n_on else S["ink"])
        fig.text(x + wdt, yy, f"{n_on} of {len(items)} in ZJX", fontsize=8.5, va="top", ha="right",
                 color=S["muted"])
        yy -= 0.036
        for e in items:
            yy = _item(fig, x, yy, wdt, e, tz)
        yy -= 0.018
    ap = next((e for e in elements if e.key == "ifr"), None)
    rows = getattr(ap, "airports", None) if ap is not None else None
    fig.text(x, yy, "AIRPORTS IFR/LIFR", fontsize=10.5, weight="bold", va="top")
    yy -= 0.028
    if ap is None or not ap.available:
        fig.text(x, yy, "n/a", fontsize=8.5, color=S["muted"], va="top")
    elif not rows:
        fig.text(x, yy, "None", fontsize=8.5, color=S["muted"], va="top")
    else:
        for name, cat, pi, pl, span in rows[:6]:
            fig.text(x, yy, name, fontsize=9, weight="bold", va="top")
            fig.text(x + 0.04, yy, f"{cat}", fontsize=9, weight="bold", va="top", color=C["ifr"])
            fig.text(x + 0.085, yy, span, fontsize=8.6, va="top", color=S["ink"])
            yy -= 0.024


def _item(fig, x, y, w, e, tz):
    rh = 0.052
    on = e.present
    fig.patches.append(FancyBboxPatch((x, y - rh + 0.006), w, rh - 0.006,
                                      boxstyle="round,pad=0,rounding_size=0.004",
                                      transform=fig.transFigure,
                                      facecolor=_tint(e.color, 0.07) if on else "#f6f7f9",
                                      edgecolor=e.color if on else S["rule"], lw=0.9 if on else 0.6))
    sw = fig.add_axes([x + 0.008, y - rh + 0.016, 0.024, rh - 0.026])
    sw.set_axis_off()
    sw.set_zorder(5)  # above the figure-level card background
    sw.add_patch(Rectangle((0.04, 0.04), 0.92, 0.92, transform=sw.transAxes, **_swatch(e)))
    tx = x + 0.042
    fig.text(tx, y - 0.009, e.name, fontsize=10, weight="bold", va="top",
             color=S["ink"] if on else S["muted"])
    if not e.available:
        status, sub = "n/a", ""
    elif on:
        status = f"{e.frac:.0%}" if e.frac >= 0.005 else "<1%"
        sub = " · ".join(x for x in (_span(e.hours, tz), getattr(e, "short", "")) if x)
    else:
        status, sub = "None", ""
    fig.text(x + w - 0.008, y - 0.009, status, fontsize=10, weight="bold", va="top", ha="right",
             color=e.color if on else S["muted"])
    if sub:
        fig.text(tx, y - 0.028, sub, fontsize=8.6, va="top", color=S["ink"])
    return y - rh


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def operational_day(spec, tz=TIMEZONE):
    """Local date of the operational day: 'today' or YYYY-MM-DD."""
    from zoneinfo import ZoneInfo

    if isinstance(spec, date):
        return spec
    return datetime.now(ZoneInfo(tz)).date()


def day_span(day, tz=TIMEZONE):
    ws = [Window.for_shift(p, day, tz) for p in TMU_PERIODS]
    return ws, ws[0].start, ws[-1].end


def run(grid, diag, mask, boundary, day, outdir, nbm=None, tz=TIMEZONE, issued=None, dpi=170, log=print,
        hrrr=None):
    from pathlib import Path

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    windows, _, _ = day_span(day, tz)
    nbm_mask = _NbmMask(nbm, boundary) if nbm is not None else None
    mask.boundary = boundary
    gfs_run = grid.meta.get("run")
    sources = []
    if hrrr is not None:
        sources.append(f"{hrrr.label} (thunderstorms)")
    if nbm is not None:
        sources.append(f"{nbm.label} (thunder probability, IFR/LIFR, LLWS, freezing rain)")
    sources.append(f"{grid.model_label}{' ' + gfs_run.replace('T', ' ') if gfs_run else ''} "
                   f"(thunder signal, turbulence FL240–FL390, icing FL025–FL210"
                   + ("" if nbm is not None else "; NBM unavailable") + ")")
    src = " · ".join(sources)
    rows = []
    for w in windows:
        s, els = analyse(w, diag, mask, nbm, nbm_mask, hrrr=hrrr)
        generalise(els, mask, diag.cfg)
        name = f"tmu_{w.shift.code}.png"
        render(w, s, els, boundary, outdir / name, src, issued, dpi, grid.synthetic)
        log(f"{w.shift.code} {w.local_label()}: " +
            (", ".join(f"{e.name} {e.frac:.0%}" for e in els if e.present) or "nothing in ZJX"))
        rows.append({
            "period": w.shift.code, "name": w.shift.name, "image": name,
            "start_utc": w.start.isoformat(), "end_utc": w.end.isoformat(),
            "local": w.local_label(), "date": w.day.isoformat(),
            "items": [{"key": e.key, "kind": e.kind, "name": e.name, "present": e.present,
                       "available": e.available, "area_frac": round(e.frac, 4),
                       "hours": _span(e.hours, w.tz), "detail": e.detail, "source": e.source}
                      for e in els],
        })
    return {"issued": (issued or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%MZ"),
            "day": day.isoformat(), "sources": src, "periods": rows}
