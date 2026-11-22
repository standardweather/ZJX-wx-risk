from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from zjxwx import gfs_aws as g

C = datetime(2026, 11, 22, 6, tzinfo=timezone.utc)


def test_plan_uses_newest_cycle_ahead_and_earlier_cycles_behind():
    p = g.plan(C - timedelta(hours=7), C + timedelta(hours=125), C)
    assert p[C - timedelta(hours=7)] == [(C - timedelta(hours=12), 5, 1.0)]
    assert p[C] == [(C - timedelta(hours=6), 6, 1.0)]          # cycle hour from the one before
    assert p[C + timedelta(hours=1)] == [(C, 1, 1.0)]
    assert p[C + timedelta(hours=120)] == [(C, 120, 1.0)]
    (c1, a, wa), (c2, b, wb) = p[C + timedelta(hours=121)]
    assert (c1, a, c2, b) == (C, 120, C, 123) and (wa, wb) == pytest.approx((2 / 3, 1 / 3))
    assert p[C + timedelta(hours=123)] == [(C, 123, 1.0)]


def test_ranges_find_every_message_and_the_running_precip_total():
    names = g._messages()
    inv, off = [], 0
    for fh_total in ("6-7 hour acc fcst", "0-7 hour acc fcst"):
        inv.append((off, f"APCP:surface:{fh_total}:")); off += 10
    for pre in names.values():
        inv.append((off, f"{pre}:7 hour fcst:")); off += 10
    inv.append((off, "TMP:2 mb:7 hour fcst:"))
    r = g._ranges(inv, 7)
    assert set(r) == set(names) | {"apcp"}
    assert r["apcp"] == (10, 19)                               # the 0-7 h total, not the 6-7 h bucket
    day = g._ranges([(0, "APCP:surface:114-120 hour acc fcst:"), (5, "APCP:surface:0-5 day acc fcst:"),
                     (9, "X")], 120)
    assert day["apcp"] == (5, 8)


def test_precip_differences_totals_and_spreads_three_hour_steps():
    data = {(C, 1): {"apcp": np.array([1.0])}, (C, 2): {"apcp": np.array([1.5])},
            (C, 120): {"apcp": np.array([10.0])}, (C, 123): {"apcp": np.array([13.0])}}
    assert g._precip(data, C, 1)[0] == 1.0
    assert g._precip(data, C, 2)[0] == 0.5
    assert g._precip(data, C, 123)[0] == 1.0
    assert g._precip(data, C, 3) is None
