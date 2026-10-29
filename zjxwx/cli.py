"""Command-line interface: ``python -m zjxwx <command> ...``"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone

from .config import SHIFTS, TIMEZONE, load_config


def _date(s):
    return s if s in ("today", "next") else date.fromisoformat(s)


def _local_day(d, tz=TIMEZONE):
    from zoneinfo import ZoneInfo

    return d if isinstance(d, date) else datetime.now(ZoneInfo(tz)).date()


def _resolve(d, shift, tz):
    """'today' = local today; 'next' = date of the next shift of this type to start."""
    from zoneinfo import ZoneInfo
    from .analysis import Window

    if isinstance(d, date):
        return d
    today = datetime.now(ZoneInfo(tz)).date()
    if d == "today":
        return today
    w = Window.for_shift(shift, today, tz)
    return today if w.start > datetime.now(timezone.utc) else today + timedelta(days=1)


def _shift(s):
    s = s.upper()
    if s not in SHIFTS:
        raise argparse.ArgumentTypeError(f"shift must be one of {', '.join(SHIFTS)}")
    return SHIFTS[s]


def _issued(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc) if s else None


def _prepare(args):
    from .analysis import Diagnostics
    from .data import load_grid
    from .geo import AirspaceMask, load_boundary

    cfg = load_config(args.config)
    grid = load_grid(args.grid)
    boundary = load_boundary(args.boundary)
    mask = AirspaceMask(boundary, grid.lats, grid.lons)
    return cfg, grid, boundary, mask, Diagnostics(grid, cfg)


def cmd_brief(args):
    from .analysis import Window, assess, summarize
    from .render import briefing_sheet

    cfg, grid, boundary, mask, diag = _prepare(args)
    day = _resolve(args.date, args.shift, args.tz)
    s = summarize(diag, Window.for_shift(args.shift, day, args.tz), mask)
    wl = assess(s, mask, cfg)
    briefing_sheet(grid, s, wl, boundary, mask, cfg, args.out, _issued(args.issued), args.dpi)
    _report(args, [(s, wl)])


def cmd_string(args):
    from .analysis import Window, assess, summarize
    from .render import shift_string

    cfg, grid, boundary, mask, diag = _prepare(args)
    sums, rates = [], []
    start = _resolve(args.start, args.shift, args.tz)
    for i in range(args.days):
        s = summarize(diag, Window.for_shift(args.shift, start + timedelta(days=i), args.tz), mask)
        sums.append(s)
        rates.append(assess(s, mask, cfg))
    shift_string(grid, sums, rates, boundary, cfg, args.out, _issued(args.issued), args.dpi)
    _report(args, list(zip(sums, rates)))


def _report(args, pairs):
    """Print a one-line summary per shift; optionally dump JSON for other tools."""
    rows = []
    for s, wl in pairs:
        print(f"{s.window.day} {s.window.shift.code}: {wl.label:8s} {wl.reason}")
        rows.append({"date": s.window.day.isoformat(), "shift": s.window.shift.code,
                     "start_utc": s.window.start.isoformat(), "end_utc": s.window.end.isoformat(),
                     "workload": wl.label, "reason": wl.reason,
                     "hazards": [{"name": h.name, "level": h.label, "detail": h.detail,
                                  "driver": h.driver} for h in wl.hazards],
                     "warnings": s.missing})
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)
    print("wrote", args.out)


def cmd_fetch(args):
    from .fetch import fetch

    gj = fetch(_local_day(args.start), args.days, args.step, model=args.model,
               log=lambda m: print(m, file=sys.stderr))
    with open(args.out, "w") as fh:
        json.dump(gj, fh, separators=(",", ":"))
    print("wrote", args.out)


def cmd_synth(args):
    from .synthetic import write

    write(args.out, _local_day(args.start), args.days, args.step)
    print("wrote", args.out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="zjxwx", description="ZJX pre-shift weather & workload briefings")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("grid", help="grid JSON from 'fetch' (or 'synth')")
        sp.add_argument("--shift", type=_shift, required=True, help="F or N")
        sp.add_argument("--out", required=True, help="output PNG (or .pdf/.svg)")
        sp.add_argument("--config", help="JSON overriding thresholds in config.DEFAULTS")
        sp.add_argument("--boundary", help="GeoJSON airspace boundary (default: bundled KZJX)")
        sp.add_argument("--tz", default=TIMEZONE, help=f"shift time zone (default {TIMEZONE})")
        sp.add_argument("--issued", help="issue time stamp, ISO 8601 (default: now)")
        sp.add_argument("--json", help="also write the workload ratings to this JSON file")
        sp.add_argument("--dpi", type=int, default=170)

    b = sub.add_parser("brief", help="one-shift briefing sheet")
    common(b)
    b.add_argument("--date", type=_date, required=True,
                   help="local date the shift starts: YYYY-MM-DD, 'today' or 'next'")
    b.set_defaults(func=cmd_brief)

    s = sub.add_parser("string", help="multi-day shift-string outlook")
    common(s)
    s.add_argument("--start", type=_date, required=True, help="YYYY-MM-DD, 'today' or 'next'")
    s.add_argument("--days", type=int, default=6)
    s.set_defaults(func=cmd_string, dpi=150)

    f = sub.add_parser("fetch", help="download GFS from Open-Meteo")
    f.add_argument("--start", type=_date, required=True, help="YYYY-MM-DD or 'today'")
    f.add_argument("--days", type=int, default=3)
    f.add_argument("--step", type=float, help="grid spacing in degrees (default 0.5)")
    f.add_argument("--model", help="Open-Meteo 'models' value, e.g. gfs025")
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_fetch)

    y = sub.add_parser("synth", help="write synthetic demo data")
    y.add_argument("--start", type=_date, required=True)
    y.add_argument("--days", type=int, default=6)
    y.add_argument("--step", type=float, default=0.5)
    y.add_argument("--out", required=True)
    y.set_defaults(func=cmd_synth)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
