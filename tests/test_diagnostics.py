"""Checks of the science core. Run with: python -m pytest tests"""
from datetime import date

import numpy as np
import pytest

from zjxwx import diagnostics as dx
from zjxwx.analysis import Window
from zjxwx.config import DEFAULTS, SHIFTS

LATS = np.arange(20.0, 40.01, 0.5)
LONS = np.arange(-95.0, -70.01, 0.5)
LON, LAT = np.meshgrid(LONS, LATS)
A = dx.EARTH_RADIUS


def test_solid_body_rotation_has_no_deformation_or_divergence():
    omega = 1e-5
    u = omega * A * np.cos(np.radians(LAT))
    v = np.zeros_like(u)
    div, st, sh = dx.kinematics(u, v, LATS, LONS)
    inner = (slice(1, -1), slice(1, -1))
    assert np.abs(div[inner]).max() < 1e-9
    assert np.abs(np.hypot(st, sh)[inner]).max() < 1e-8  # vs. 1e-5 vorticity scale


def test_pure_shear_flow_deformation_equals_shear():
    # u = k*y (in metres) -> shearing deformation k plus the small u*tan(phi)/a term
    k = 2e-5
    y = np.radians(LAT - 30.0) * A
    u, v = k * y, np.zeros_like(y)
    _, st, sh = dx.kinematics(u, v, LATS, LONS)
    j = list(LATS).index(30.0)
    assert sh[j, 10] == pytest.approx(k, rel=1e-3)
    assert abs(st[j, 10]) < 1e-12


def test_original_shearing_term_bug_is_fixed():
    # Pure dv/dy stretching: the original (dVdx + dVdy) would report shear
    k = 1e-5
    y = np.radians(LAT - 30.0) * A
    _, st, sh = dx.kinematics(np.zeros_like(y), k * y, LATS, LONS)
    j = list(LATS).index(30.0)
    assert abs(sh[j, 10]) < 1e-12
    assert abs(st[j, 10]) == pytest.approx(k, rel=0.05)


def test_icing_interest_prefers_saturated_minus_six_over_dry_minus_seventeen():
    cfg = DEFAULTS["icing"]
    t = lambda x: dx._interp_table(x, cfg["t_interest"])
    rh = lambda x: dx._interp_table(x, cfg["rh_interest"])
    assert t(-6) * rh(97) == pytest.approx(1.0)
    assert t(-17) * rh(76) < 0.25
    assert t(2) == 0 and t(-30) == 0


def test_shift_window_handles_dst_and_hour_ending_precip():
    w = Window.for_shift(SHIFTS["F"], date(2026, 10, 30))      # EDT
    assert (w.start.hour, w.start.minute) == (9, 15)
    acc = w.accum_hours()
    assert acc[0].hour == 10 and acc[-1].hour == 18            # covers 0915-1715Z
    w2 = Window.for_shift(SHIFTS["F"], date(2026, 11, 2))      # EST
    assert w2.start.hour == 10
    n = Window.for_shift(SHIFTS["N"], date(2026, 11, 2))
    assert n.end.day == 3 and n.end.hour == 2                  # crosses 00Z
