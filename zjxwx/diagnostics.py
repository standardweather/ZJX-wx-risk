"""Aviation hazard diagnostics from GFS pressure-level fields.

All functions are vectorised over leading (time) axes; the last two axes
are (lat, lon) on a regular latitude/longitude grid.

References
----------
Ellrod, G. P. and D. I. Knapp, 1992: An objective clear-air turbulence
    forecasting technique: verification and operational use.
    Wea. Forecasting, 7, 150-165.
Ellrod, G. P. and J. A. Knox, 2010: Improvements to an operational
    clear-air turbulence diagnostic index by addition of a divergence
    trend term. Wea. Forecasting, 25, 789-798.
Bernstein, B. C., et al., 2005: Current Icing Potential: algorithm
    description and comparison with aircraft observations.
    J. Appl. Meteor., 44, 969-986.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

EARTH_RADIUS = 6.371e6  # m


# --------------------------------------------------------------------------
# Kinematics on the sphere
# --------------------------------------------------------------------------
def _grad(f, lats, lons):
    """Return (df/dx, df/dy) in units of f per metre."""
    phi = np.radians(lats)
    lam = np.radians(lons)
    dfdphi = np.gradient(f, phi, axis=-2)
    dfdlam = np.gradient(f, lam, axis=-1)
    dfdx = dfdlam / (EARTH_RADIUS * np.cos(phi)[:, None])
    dfdy = dfdphi / EARTH_RADIUS
    return dfdx, dfdy


def kinematics(u, v, lats, lons):
    """Divergence, stretching and shearing deformation (s^-1) on the sphere.

    Includes the curvature terms (u, v) * tan(phi) / a, which are the same
    order as the deformation itself in a 50 m/s jet at 30N and are needed for
    solid-body rotation to give zero deformation.
    """
    dudx, dudy = _grad(u, lats, lons)
    dvdx, dvdy = _grad(v, lats, lons)
    tan_a = (np.tan(np.radians(lats)) / EARTH_RADIUS)[:, None]
    div = dudx + dvdy - v * tan_a
    stretch = dudx - dvdy - v * tan_a
    shear = dvdx + dudy + u * tan_a
    return div, stretch, shear


def total_deformation(u, v, lats, lons):
    _, dst, dsh = kinematics(u, v, lats, lons)
    return np.hypot(dst, dsh)


# --------------------------------------------------------------------------
# Clear-air turbulence: Ellrod TI1 and Ellrod-Knox DTI
# --------------------------------------------------------------------------
def _smooth(a, n):
    """n x n box average over the horizontal axes (n <= 1: unchanged)."""
    if n <= 1:
        return a
    size = (1,) * (a.ndim - 2) + (n, n)
    return uniform_filter(np.nan_to_num(a), size=size, mode="nearest")


def ellrod_layers(grid, levels, times=slice(None), smooth=3):
    """Ellrod TI1 (1e-7 s^-2) and layer-mean divergence for each layer.

    Each layer is bounded by consecutive pressure levels in ``levels``
    (e.g. 400-350, 350-300, ...). VWS = |V_top - V_bot| / (Z_top - Z_bot)
    and DEF uses the layer-mean wind, as in Ellrod & Knapp (1992). Layers
    with missing or non-positive thickness are left as NaN rather than
    clamped, so bad data cannot masquerade as a turbulence signal.

    TI and divergence are box-smoothed over ``smooth`` x ``smooth`` points.
    Hourly GFS divergence at 0.5 deg carries grid-scale noise (convective
    and gravity-wave signals) that the 40-80 km grids Ellrod's thresholds
    were developed on did not resolve.

    Returns a list of dicts: {"bot", "top", "ti", "div", "vws_kt_kft"}.
    """
    out = []
    for bot, top in zip(levels[:-1], levels[1:]):
        wb, wt = grid.wind(bot), grid.wind(top)
        zb = grid.get(f"geopotential_height_{bot}hPa")
        zt = grid.get(f"geopotential_height_{top}hPa")
        if wb is None or wt is None or zb is None or zt is None:
            continue
        ub, vb = (a[times] for a in wb)
        ut, vt = (a[times] for a in wt)
        dz = zt[times] - zb[times]
        dz = np.where(dz > 0, dz, np.nan)
        vws = np.hypot(ut - ub, vt - vb) / dz
        um, vm = 0.5 * (ub + ut), 0.5 * (vb + vt)
        div, dst, dsh = kinematics(um, vm, grid.lats, grid.lons)
        ti = np.hypot(dst, dsh) * vws * 1e7
        ti = np.where(np.isnan(ti), np.nan, _smooth(ti, smooth))
        out.append({"bot": bot, "top": top, "ti": ti, "div": _smooth(div, smooth),
                    # kt per 1000 ft: rule of thumb >= 6 suggests moderate CAT
                    "vws_kt_kft": vws * 1.943844 * 304.8})
    return out


def convective_mask(precip, rate=2.0, hours=3):
    """True where the model has convective-scale precip (>= rate mm/h) within
    the past ``hours`` at the point or a neighbour."""
    p = np.nan_to_num(precip)
    m = maximum_filter(p, size=(hours + 1, 3, 3), origin=(hours // 2, 0, 0), mode="nearest")
    return m >= rate


def ellrod_knox_dti(layers, dvt_c=0.1, lag_steps=6, dvt_hours=6.0, div_scale=1e6, convective=None):
    """Divergence-modified turbulence index DTI = TI + DVT per layer.

    DVT = C * (DIV(t) - DIV(t - dt)) with divergence multiplied by
    ``div_scale`` (i.e. expressed in 1e-6 s^-1). Ellrod & Knox (2010) set
    C = 0.1 for a 6 h interval so that DVT is comparable to TI only when
    divergence changes are large, and note C should scale with dt. The paper
    does not state the divergence units; 1e-6 s^-1 meets that intent on real
    GFS data (DVT p50 ~1, p99 ~10 vs TI p50 ~1, p99 ~15), whereas 1e-7 s^-1
    made DVT ten times too large and flagged all of ZJX every day.

    Both terms are clear-air diagnostics. Near model convection the upper
    winds are dominated by thunderstorm outflow (divergence ~1e-4 s^-1,
    direction reversals over 1 km), which produced DTI values of 40-100 on
    real GFS data. Where ``convective`` (boolean (nt, ny, nx)) is set the
    index is zeroed, and that turbulence is carried by the thunderstorm
    hazard instead.
    Needs ``lag_steps`` earlier time steps; where unavailable DVT = 0 and
    DTI falls back to TI1. Negative values are set to zero, as in the paper.

    Returns (dti_max, ti_max, layer_of_max) over layers, each (nt, ny, nx).
    """
    c = dvt_c * dvt_hours / 6.0
    dtis, tis = [], []
    for L in layers:
        div = L["div"] * div_scale
        dvt = np.zeros_like(div)
        if div.shape[0] > lag_steps:
            dvt[lag_steps:] = c * (div[lag_steps:] - div[:-lag_steps])
        dti = np.clip(np.nan_to_num(L["ti"] + dvt, nan=-1.0), 0, None)
        if convective is not None:
            dti[convective] = 0.0
        dtis.append(dti)
        tis.append(L["ti"])
    dti = np.stack(dtis)
    ti = np.stack(tis)
    which = np.argmax(dti, axis=0)
    return dti.max(axis=0), np.nanmax(ti, axis=0), which


# --------------------------------------------------------------------------
# Icing
# --------------------------------------------------------------------------
def _interp_table(x, table):
    xs, ys = np.array(table, dtype=float).T
    order = np.argsort(xs)
    return np.interp(x, xs[order], ys[order], left=ys[order][0], right=ys[order][-1])


def icing_potential(grid, levels, cfg, times=slice(None)):
    """CIP/FIP-style icing potential (0..1) per level and its column maximum.

    Potential = temperature interest x moisture interest. The temperature
    map peaks between -3 and -12 C (where supercooled liquid water is most
    common) and tapers to zero at -25 C (glaciation). Moisture interest
    ramps from RH 70 % to 95 % as a cloud proxy. This replaces the original
    binary "-18 <= T <= 0 and RH >= 75" test, which gave equal weight to
    a dry -17 C layer and a saturated -6 C layer.

    Returns (column_max (nt,ny,nx), per_level dict, base_fl, top_fl) where
    base/top are the lowest/highest flight level with potential >= threshold.
    """
    from .config import LEVEL_FL

    thr = cfg["potential_threshold"]
    per = {}
    for p in levels:
        t = grid.get(f"temperature_{p}hPa")
        rh = grid.get(f"relative_humidity_{p}hPa")
        if t is None or rh is None:
            continue
        pot = _interp_table(t[times], cfg["t_interest"]) * _interp_table(rh[times], cfg["rh_interest"])
        per[p] = np.where(np.isnan(t[times]) | np.isnan(rh[times]), 0.0, pot)
    if not per:
        raise ValueError("no temperature/RH levels available for icing")
    stack = np.stack([per[p] for p in per])
    fls = np.array([LEVEL_FL.get(p, np.nan) for p in per], dtype=float)
    hit = stack >= thr
    fl_b = np.where(hit, fls[:, None, None, None], np.inf).min(axis=0)
    fl_t = np.where(hit, fls[:, None, None, None], -np.inf).max(axis=0)
    return stack.max(axis=0), per, np.where(np.isinf(fl_b), np.nan, fl_b), np.where(np.isinf(fl_t), np.nan, fl_t)


def freezing_precip(grid, cfg, times=slice(None)):
    """Boolean freezing-rain/drizzle signal (SLD icing at low altitude).

    Surface temperature at or below freezing, measurable precipitation and a
    warm (> +1 C) layer at 850 or 700 hPa: the classic warm-nose profile.
    """
    t2 = grid.get("temperature_2m")
    pr = grid.get("precipitation")
    if t2 is None or pr is None:
        return None
    warm = np.full(t2[times].shape, False)
    for p in (925, 850, 700):
        t = grid.get(f"temperature_{p}hPa")
        if t is not None:
            warm |= np.nan_to_num(t[times], nan=-99) > cfg["fzra_warm_nose"]
    return (t2[times] <= cfg["fzra_t2m"]) & (pr[times] >= cfg["fzra_precip"]) & warm


def freezing_level_fl(grid, times=slice(None)):
    """Freezing level as a flight level (hundreds of ft MSL)."""
    z = grid.get("freezing_level_height")
    if z is not None:
        return z[times] * 3.28084 / 100.0
    return None


# --------------------------------------------------------------------------
# Convection
# --------------------------------------------------------------------------
def thunderstorm_hours(grid, cfg, times=slice(None)):
    """Boolean (nt, ny, nx): hour with a model thunderstorm signal.

    Requires all of: CAPE >= cape_min, lifted index <= li_max (if
    available), CIN >= cin_min (if available) and model precipitation
    >= precip_min mm/h (the model is actually convecting, not just unstable).
    """
    cape = grid.get("cape")
    pr = grid.get("precipitation")
    if cape is None or pr is None:
        return None
    ok = (np.nan_to_num(cape[times]) >= cfg["cape_min"]) & (np.nan_to_num(pr[times]) >= cfg["precip_min"])
    li = grid.get("lifted_index")
    if li is not None:
        ok &= np.nan_to_num(li[times], nan=99) <= cfg["li_max"]
    cin = grid.get("convective_inhibition")
    if cin is not None:
        # sign conventions differ between sources; compare the magnitude
        ok &= -np.abs(np.nan_to_num(cin[times], nan=0)) >= cfg["cin_min"]
    return ok
