from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from zjxwx import noaa_aws as g

C = datetime(2026, 11, 22, 6, tzinfo=timezone.utc)
H = timedelta(hours=1)


def test_gfs_plan_uses_newest_cycle_ahead_and_earlier_cycles_behind():
    p = g.plan(g.GFS, C - 7 * H, C + 125 * H, C)
    assert p[C - 7 * H] == [(C - 12 * H, 5, 1.0)]
    assert p[C] == [(C - 6 * H, 6, 1.0)]          # cycle hour from the one before
    assert p[C + H] == [(C, 1, 1.0)]
    assert p[C + 120 * H] == [(C, 120, 1.0)]
    (c1, a, wa), (c2, b, wb) = p[C + 121 * H]
    assert (c1, a, c2, b) == (C, 120, C, 123) and (wa, wb) == pytest.approx((2 / 3, 1 / 3))
    assert p[C + 123 * H] == [(C, 123, 1.0)]


def test_hrrr_plan_uses_hourly_runs_behind_and_stops_at_f48():
    p = g.plan(g.HRRR, C - 7 * H, C + 50 * H, C)
    assert p[C - 7 * H] == [(C - 8 * H, 1, 1.0)]
    assert p[C] == [(C - H, 1, 1.0)]
    assert p[C + 48 * H] == [(C, 48, 1.0)]
    assert p[C + 49 * H] == []


def _inv(model, fh, apcp):
    inv, off = [], 0
    for desc in apcp:
        inv.append((off, f"APCP:surface:{desc}:")); off += 10
    for pre in g._messages(model).values():
        inv.append((off, f"{pre}:{fh} hour fcst:")); off += 10
    inv.append((off, "TMP:2 mb:x:"))
    return inv


def test_ranges_find_every_message_and_the_right_precip_record():
    r = g._ranges(g.GFS, _inv(g.GFS, 7, ["6-7 hour acc fcst", "0-7 hour acc fcst"]), 7)
    assert set(r) == set(g._messages(g.GFS)) | {"apcp"}
    assert r["apcp"] == (10, 19)                   # GFS: the 0-7 h total, not the 6-7 h bucket
    r = g._ranges(g.HRRR, _inv(g.HRRR, 12, ["0-12 hour acc fcst", "11-12 hour acc fcst"]), 12)
    assert set(r) == set(g._messages(g.HRRR)) | {"apcp"}
    assert r["apcp"] == (10, 19)                   # HRRR: the hourly amount
    assert "LFTX:500-1000 mb" in g._messages(g.HRRR).values()
    day = g._ranges(g.GFS, [(0, "APCP:surface:114-120 hour acc fcst:"), (5, "APCP:surface:0-5 day acc fcst:"),
                            (9, "X")], 120)
    assert day["apcp"] == (5, 8)


def test_gfs_precip_differences_totals_and_spreads_three_hour_steps():
    data = {(C, 1): {"apcp": np.array([1.0])}, (C, 2): {"apcp": np.array([1.5])},
            (C, 120): {"apcp": np.array([10.0])}, (C, 123): {"apcp": np.array([13.0])}}
    assert g._precip(g.GFS, data, C, 1)[0] == 1.0
    assert g._precip(g.GFS, data, C, 2)[0] == 0.5
    assert g._precip(g.GFS, data, C, 123)[0] == 1.0
    assert g._precip(g.GFS, data, C, 3) is None
    assert g._precip(g.HRRR, {(C, 5): {"apcp": np.array([0.7])}}, C, 5)[0] == 0.7
