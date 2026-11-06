"""OM/TMU/OS periods and map pipeline. Run with: python -m pytest tests"""
from datetime import date, datetime, timedelta, timezone

import numpy as np

from zjxwx import nbm as nb
from zjxwx.config import TMU_PERIODS
from zjxwx.tmu import _span, day_span


def test_periods_tile_the_operational_day_across_dst():
    for day in (date(2026, 7, 1), date(2026, 11, 1), date(2026, 3, 8)):
        ws, t0, t1 = day_span(day)
        assert [w.shift.code for w in ws] == [p.code for p in TMU_PERIODS]
        for a, b in zip(ws, ws[1:]):
            assert a.end == b.start
        assert ws[0].start.astimezone(ws[0].tz).hour == 5
        assert ws[-1].end.astimezone(ws[-1].tz).hour == 5
        assert ws[-1].end.astimezone(ws[-1].tz).date() == day + timedelta(days=1)


def test_span_merges_consecutive_hours():
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/New_York")
    t = datetime(2026, 11, 6, 15, tzinfo=timezone.utc)  # 1000 EST
    hrs = [(t + timedelta(hours=k), t + timedelta(hours=k)) for k in (0, 1, 2, 5)]
    assert _span(hrs, tz) == "1000–1200L, 1500L"
    assert _span([], tz) == ""


def test_nbm_npz_round_trip(tmp_path):
    lats = np.linspace(25, 35, 12, dtype="float32").reshape(3, 4)
    lons = np.linspace(-88, -77, 12, dtype="float32").reshape(3, 4)
    times = np.array(["2026-11-06T10:00", "2026-11-06T11:00"], dtype="datetime64[m]")
    f = {n: np.random.rand(2, 3, 4).astype("float32") for n in nb.FIELDS}
    g = nb.NbmGrid(lats, lons, times, f, {"cycle": "2026-11-06 04Z"})
    g.save(tmp_path / "n.npz")
    h = nb.NbmGrid.load(tmp_path / "n.npz")
    assert h.meta["cycle"] == "2026-11-06 04Z"
    assert (h.times == times).all()
    assert h.index(datetime(2026, 11, 6, 11, tzinfo=timezone.utc)) == 1
    np.testing.assert_array_equal(h.get("tstm"), f["tstm"])


def test_parse_cycle():
    assert nb.parse_cycle("2026110604") == datetime(2026, 11, 6, 4, tzinfo=timezone.utc)
    assert nb.parse_cycle("2026-11-06T04Z") == datetime(2026, 11, 6, 4, tzinfo=timezone.utc)


def test_tmu_renders_from_synthetic_gfs_only(tmp_path):
    from zjxwx.cli import main
    import json

    grid = tmp_path / "g.json"
    main(["synth", "--start", "2026-10-30", "--days", "2", "--out", str(grid)])
    main(["tmu", str(grid), "--date", "2026-10-30", "--outdir", str(tmp_path / "t"), "--dpi", "40"])
    res = json.loads((tmp_path / "t" / "tmu.json").read_text())
    assert [p["period"] for p in res["periods"]] == ["AM", "PM", "MID"]
    for p in res["periods"]:
        assert (tmp_path / "t" / p["image"]).stat().st_size > 10_000
        kinds = [i["kind"] for i in p["items"]]
        assert kinds.count("HAZARD") == 3 and kinds.count("AWARENESS") == 4
        ifr = next(i for i in p["items"] if i["key"] == "ifr")
        assert not ifr["available"]  # no NBM file
