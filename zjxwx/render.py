"""Briefing graphics: single-shift briefing sheet and multi-day shift string."""
from __future__ import annotations

from datetime import datetime, timezone

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import box

from .config import AIRPORTS, CATEGORY, DISCLAIMER, EXTENT, STYLE as S
from .geo import basemap

PROJ = ccrs.LambertConformal(central_longitude=-82.2, central_latitude=29.0,
                             standard_parallels=(25.0, 33.0))
PC = ccrs.PlateCarree()

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 9,
    "text.color": S["ink"],
    "axes.edgecolor": S["rule"],
    "hatch.linewidth": 0.6,
})

HALO = [pe.withStroke(linewidth=2.6, foreground="white")]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _fine(grid, field, res=0.1):
    """Linear upsample for display so contours are not stair-stepped.

    Purely cosmetic: it adds no information, and statistics are computed
    on the model grid via AirspaceMask, not on this field.
    """
    fy = np.linspace(grid.lats[0], grid.lats[-1], int(round(np.ptp(grid.lats) / res)) + 1)
    fx = np.linspace(grid.lons[0], grid.lons[-1], int(round(np.ptp(grid.lons) / res)) + 1)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    f = RegularGridInterpolator((grid.lats, grid.lons), np.asarray(field, float))((FY, FX))
    return FX, FY, f


def _cat_color(level):
    return S["cat"][CATEGORY[level]]


def _tint(color, a=0.16):
    r, g, b, _ = to_rgba(color)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def _badge(fig, x, y, w, h, level, title="FORECAST WORKLOAD", big=26):
    c = _cat_color(level)
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006",
                                      transform=fig.transFigure, facecolor=c, edgecolor="none"))
    fig.text(x + w / 2, y + h * 0.74, title, ha="center", va="center", color="white",
             fontsize=8, weight="semibold", alpha=0.9)
    fig.text(x + w / 2, y + h * 0.36, CATEGORY[level], ha="center", va="center", color="white",
             fontsize=big, weight="bold")


def _header(fig, kicker, title, sub, workload=None, height=0.085):
    top = 1.0
    fig.patches.append(Rectangle((0, top - height), 1, height, transform=fig.transFigure,
                                 facecolor=S["navy"], edgecolor="none", zorder=-1))
    fig.text(0.025, top - height * 0.22, kicker, color="#9fb3d1", fontsize=9,
             weight="semibold", va="center")
    fig.text(0.025, top - height * 0.52, title, color="white", fontsize=21, weight="bold", va="center")
    fig.text(0.025, top - height * 0.82, sub, color="#d8e1ee", fontsize=10.5, va="center")
    if workload is not None:
        _badge(fig, 0.80, top - height + height * 0.14, 0.175, height * 0.72, workload)
    return top - height


def _meta_line(fig, y, grid, issued):
    run = grid.meta.get("run")
    bits = [f"Model: {grid.model_label}"]
    if run:
        bits.append(f"Run: {run.replace('T', ' ')}")
    step = grid.meta.get("step")
    if step:
        bits.append(f"Grid: {step:g}°")
    bits.append(f"Issued: {issued:%Y-%m-%d %H%MZ}")
    fig.text(0.025, y, "   ·   ".join(bits), fontsize=8.5, color=S["muted"], va="center")
    if grid.synthetic:
        fig.text(0.975, y, "SYNTHETIC DEMO DATA — NOT A FORECAST", fontsize=8.5, color="white",
                 weight="bold", ha="right", va="center",
                 bbox=dict(facecolor="#b3261e", edgecolor="none", boxstyle="round,pad=0.35"))


def _footer(fig, extra=""):
    fig.add_artist(Line2D([0.025, 0.975], [0.042, 0.042], transform=fig.transFigure,
                          color=S["rule"], lw=0.8))
    text = DISCLAIMER + (f"  {extra}" if extra else "")
    fig.text(0.025, 0.022, text, fontsize=7, color=S["muted"], va="center", wrap=True)


# --------------------------------------------------------------------------
# Basemap
# --------------------------------------------------------------------------
def _map_axes(fig, rect):
    ax = fig.add_axes(rect, projection=PROJ)
    ax.set_extent(EXTENT, crs=PC)
    ax.set_facecolor(S["ocean"])
    ax.spines["geo"].set_edgecolor(S["rule"])
    ax.spines["geo"].set_linewidth(0.8)
    layers = basemap()
    ax.add_geometries(layers.get("land", []), PC, facecolor=S["land"], edgecolor="none", zorder=0)
    ax.add_geometries(layers.get("lake", []), PC, facecolor=S["ocean"], edgecolor="none", zorder=0.5)
    return ax


def _overlay_geography(ax, boundary, airports=True, label_size=7.5):
    layers = basemap()
    ax.add_geometries(layers.get("state", []), PC, facecolor="none", edgecolor=S["state"],
                      lw=0.5, zorder=4)
    ax.add_geometries(layers.get("country", []), PC, facecolor="none", edgecolor=S["coast"],
                      lw=0.6, zorder=4)
    ax.add_geometries(layers.get("land", []), PC, facecolor="none", edgecolor=S["coast"],
                      lw=0.6, zorder=4)
    # De-emphasise everything outside ZJX so the eye lands on our airspace.
    outside = box(-100, 15, -65, 45).difference(boundary)
    ax.add_geometries([outside], PC, facecolor="white", alpha=0.42, edgecolor="none", zorder=5)
    ax.add_geometries([boundary], PC, facecolor="none", edgecolor="white", lw=3.6, zorder=6)
    ax.add_geometries([boundary], PC, facecolor="none", edgecolor=S["boundary"], lw=1.6, zorder=7)
    if not airports:
        return
    for name, (x, y) in AIRPORTS.items():
        inside = boundary.contains(box(x - 1e-3, y - 1e-3, x + 1e-3, y + 1e-3))
        col = S["ink"] if inside else "#8a93a3"
        ax.plot(x, y, "o", ms=3.2, mfc=col, mec="white", mew=0.7, transform=PC, zorder=8)
        ax.text(x + 0.13, y + 0.08, name, fontsize=label_size, weight="semibold", color=col,
                transform=PC, zorder=8, path_effects=HALO)


def _panel_title(fig, ax, letter, title, sub):
    bb = ax.get_position()
    fig.text(bb.x0, bb.y1 + 0.03, f"{letter}  {title}", fontsize=11.5, weight="bold", va="bottom")
    fig.text(bb.x0, bb.y1 + 0.008, sub, fontsize=8.2, color=S["muted"], va="bottom")


def _legend_row(fig, ax, handles, ncol=None, y_off=0.012):
    bb = ax.get_position()
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(bb.x0 - 0.004, bb.y0 - y_off),
               ncol=ncol or len(handles), frameon=False, fontsize=7.6, handlelength=1.6,
               handleheight=1.1, columnspacing=1.0, borderaxespad=0, handletextpad=0.5)


def _colorbar(fig, ax, mappable, label, ticks, fmt="{:g}", width=0.62):
    bb = ax.get_position()
    cax = fig.add_axes([bb.x0, bb.y0 - 0.032, bb.width * width, 0.011])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.2, length=0, pad=2)
    cb.ax.set_xticklabels([fmt.format(t) for t in ticks])
    if label:
        cax.text(1.06, 0.5, label, transform=cax.transAxes, fontsize=7.6, va="center", color=S["muted"])
    return cb


# --------------------------------------------------------------------------
# Map layers
# --------------------------------------------------------------------------
def _precip_cmap():
    cols = S["precip_colors"]
    return ListedColormap(cols), BoundaryNorm(S["precip_levels_in"], len(cols))


def layer_precip_ts(ax, grid, s, compact=False):
    X, Y, P = _fine(grid, s.precip_in)
    cmap, norm = _precip_cmap()
    lv = S["precip_levels_in"]
    cf = ax.contourf(X, Y, P, levels=lv, cmap=cmap, norm=norm, extend="max", transform=PC, zorder=2)
    cf.cmap.set_over(S["precip_colors"][-1])
    if s.ts_hours is not None and np.nanmax(s.ts_hours) >= 1:
        X, Y, T = _fine(grid, s.ts_hours)
        ax.contourf(X, Y, T, levels=[2.5, 1e3], colors="none", hatches=["...."], transform=PC, zorder=3)
        ax.contour(X, Y, T, levels=[0.5, 2.5], colors=S["ts"], linewidths=[0.9, 1.7],
                   transform=PC, zorder=3)
    return cf


def layer_turbulence(ax, grid, s, levels, compact=False):
    X, Y, D = _fine(grid, s.dti_max)
    lv = list(levels) + [1e3]
    cf = ax.contourf(X, Y, D, levels=lv, colors=S["turb"], alpha=0.85, transform=PC, zorder=2)
    ax.contour(X, Y, D, levels=levels, colors=[S["turb"][1], S["turb"][2], S["turb"][2]],
               linewidths=[0.5, 0.9, 1.4], transform=PC, zorder=2.5)
    if s.jet_u is not None and not compact:
        spd = np.hypot(s.jet_u, s.jet_v) * 1.943844
        X2, Y2, J = _fine(grid, spd)
        cs = ax.contour(X2, Y2, J, levels=[70, 90, 110, 130, 150], colors=S["jet"],
                        linewidths=[0.7, 0.9, 1.1, 1.3, 1.5], linestyles="--", transform=PC, zorder=3)
        ax.clabel(cs, fmt="%d kt", fontsize=6.8, inline=True, inline_spacing=2)
        st = 2 if grid.lats.size > 16 else 1
        sl = (slice(1, None, st), slice(1, None, st))
        LON, LAT = np.meshgrid(grid.lons, grid.lats)
        # Barbs only where the jet matters (>= 50 kt) to keep the panel readable.
        ub, vb = s.jet_u[sl] * 1.943844, s.jet_v[sl] * 1.943844
        keep = np.hypot(ub, vb) >= 50
        ax.barbs(LON[sl][keep], LAT[sl][keep], ub[keep], vb[keep], length=4.6,
                 linewidth=0.45, color="#4a4f5c", transform=PC, zorder=3.5, alpha=0.75)
    return cf


def layer_icing(ax, grid, s, compact=False):
    X, Y, H = _fine(grid, s.ice_hours)
    blues = ["#d6e6f7", "#8fbbe6", "#3f86cc", "#1453a3"]
    cf = ax.contourf(X, Y, H, levels=[0.5, 3, 6, 9, 1e3], colors=blues, alpha=0.9, transform=PC, zorder=2)
    if s.fzra_hours is not None and np.nanmax(s.fzra_hours) >= 1:
        X2, Y2, F = _fine(grid, s.fzra_hours)
        ax.contourf(X2, Y2, F, levels=[0.5, 1e3], colors="none", hatches=["xxx"], transform=PC, zorder=3)
        ax.contour(X2, Y2, F, levels=[0.5], colors="#8e24aa", linewidths=1.2, transform=PC, zorder=3)
    if s.fzl_fl is not None and not compact:
        X3, Y3, Z = _fine(grid, s.fzl_fl)
        lo = int(np.nanmin(Z) // 20 * 20)
        lv = np.arange(max(lo, 0), np.nanmax(Z) + 20, 20)
        cs = ax.contour(X3, Y3, Z, levels=lv, colors=S["fzlvl"], linewidths=0.8,
                        linestyles=[(0, (4, 2))], transform=PC, zorder=3)
        ax.clabel(cs, fmt=lambda v: f"FL{v:03.0f}", fontsize=6.8, inline=True)
    return cf


# --------------------------------------------------------------------------
# Single-shift briefing sheet
# --------------------------------------------------------------------------
def briefing_sheet(grid, s, wl, boundary, mask, cfg, out, issued=None, dpi=170):
    issued = issued or datetime.now(timezone.utc)
    w = s.window
    fig = plt.figure(figsize=(16, 11), facecolor=S["paper"])
    y = _header(fig, "ZJX ARTCC  ·  PRE-SHIFT WEATHER & WORKLOAD BRIEFING",
                f"{w.shift.code} shift ({w.shift.name.lower()})  —  {w.day:%a %d %b %Y}",
                f"{w.local_label()}   ·   {w.utc_label()}", wl.level)
    _meta_line(fig, y - 0.018, grid, issued)

    mw, mh, gap, x0, my = 0.305, 0.43, 0.0275, 0.025, 0.405
    axA = _map_axes(fig, [x0, my, mw, mh])
    axB = _map_axes(fig, [x0 + mw + gap, my, mw, mh])
    axC = _map_axes(fig, [x0 + 2 * (mw + gap), my, mw, mh])

    cfA = layer_precip_ts(axA, grid, s)
    _overlay_geography(axA, boundary)
    _panel_title(fig, axA, "A", "Precipitation & thunderstorms",
                 "Total precip over the shift (inches) · model thunderstorm hours")
    _colorbar(fig, axA, cfA, "", S["precip_levels_in"][:-1], "{:g}", width=0.5)
    bb = axA.get_position()
    fig.legend(handles=[Line2D([], [], color=S["ts"], lw=0.9, label="TS ≥ 1 h"),
                        Patch(facecolor="white", edgecolor=S["ts"], hatch="....", lw=1.6, label="TS ≥ 3 h")],
               loc="upper right", bbox_to_anchor=(bb.x1, bb.y0 - 0.014), ncol=2, frameon=False,
               fontsize=7.6, handlelength=1.6, borderaxespad=0, columnspacing=1.0)

    tl = cfg["turbulence"]["levels"]
    cfB = layer_turbulence(axB, grid, s, tl)
    _overlay_geography(axB, boundary)
    _panel_title(fig, axB, "B", "Clear-air turbulence, FL240–FL390",
                 f"Max Ellrod–Knox DTI over shift · 250 hPa jet at {s.jet_time:%H%MZ}")
    _legend_row(fig, axB, [
        Patch(facecolor=S["turb"][0], label=f"DTI {tl[0]}–{tl[1]}  light–mod"),
        Patch(facecolor=S["turb"][1], label=f"{tl[1]}–{tl[2]}  moderate"),
        Patch(facecolor=S["turb"][2], label=f"≥ {tl[2]}  mod–severe"),
        Line2D([], [], color=S["jet"], lw=1, ls="--", label="Isotachs (kt)")], ncol=4)

    cfC = layer_icing(axC, grid, s)
    _overlay_geography(axC, boundary)
    _panel_title(fig, axC, "C", "Icing potential, 2,500–21,000 ft",
                 "Hours with supercooled-liquid potential ≥ 0.5 · freezing level")
    blues = ["#d6e6f7", "#8fbbe6", "#3f86cc", "#1453a3"]
    _legend_row(fig, axC, [Patch(facecolor=c, label=l) for c, l in
                           zip(blues, ["1–2 h", "3–5 h", "6–8 h", "9+ h"])] +
                [Line2D([], [], color=S["fzlvl"], lw=1, ls=(0, (4, 2)), label="Freezing lvl"),
                 Patch(facecolor="white", edgecolor="#8e24aa", hatch="xxx", label="FZRA")], ncol=6)

    _scorecard(fig, [0.025, 0.07, 0.555, 0.225], wl)
    _timeline(fig, [0.635, 0.1, 0.34, 0.175], s)

    _footer(fig, "Method notes: README / CRITIQUE in the zjx-wx package.")
    fig.savefig(out, dpi=dpi, facecolor=S["paper"])
    plt.close(fig)
    return out


def _scorecard(fig, rect, wl):
    x, y, w, h = rect
    fig.text(x, y + h + 0.012, "Workload drivers inside ZJX", fontsize=11.5, weight="bold")
    fig.text(x, y + h - 0.006, f"Overall {wl.label}: {wl.reason}", fontsize=8.4, color=S["muted"])
    n = len(wl.hazards)
    rh = (h - 0.03) / n
    for i, hz in enumerate(wl.hazards):
        yy = y + h - 0.03 - (i + 1) * rh
        c = _cat_color(hz.level)
        fig.patches.append(FancyBboxPatch((x, yy + 0.006), w, rh - 0.008,
                                          boxstyle="round,pad=0,rounding_size=0.004",
                                          transform=fig.transFigure, facecolor=_tint(c, 0.08),
                                          edgecolor=S["rule"], lw=0.6))
        fig.patches.append(FancyBboxPatch((x + 0.007, yy + 0.006 + rh * 0.2), 0.072, rh * 0.5,
                                          boxstyle="round,pad=0,rounding_size=0.004",
                                          transform=fig.transFigure, facecolor=c, edgecolor="none"))
        fig.text(x + 0.043, yy + 0.006 + rh * 0.45, hz.label, ha="center", va="center",
                 color="white", fontsize=8.6, weight="bold")
        fig.text(x + 0.092, yy + rh * 0.62, hz.name, fontsize=10, weight="semibold", va="center")
        fig.text(x + 0.092, yy + rh * 0.30, hz.detail, fontsize=8.2, color=S["muted"], va="center")


def _timeline(fig, rect, s):
    ax = fig.add_axes(rect)
    fig.text(rect[0], rect[1] + rect[3] + 0.037, "Hazard coverage through the shift",
             fontsize=11.5, weight="bold")
    fig.text(rect[0], rect[1] + rect[3] + 0.019, "% of ZJX airspace affected each hour (local time)",
             fontsize=8.4, color=S["muted"])
    tz = s.window.tz
    xs = np.arange(len(s.hours))
    colors = {"Thunderstorms": S["ts"], "Icing": S["ice"]}
    ymax = 10
    for name, series in s.series.items():
        if series is None:
            continue
        c = colors.get(name, S["turb"][1])
        ax.plot(xs, series * 100, color=c, lw=2, solid_capstyle="round", label=name)
        ax.plot(xs[-1], series[-1] * 100, "o", ms=4, color=c, mec="white", mew=1)
        ymax = max(ymax, float(np.nanmax(series)) * 100)
    ax.set_ylim(0, min(100, ymax * 1.15 + 2))
    ax.set_xlim(-0.3, xs[-1] + 0.3)
    step = 2 if len(xs) > 8 else 1
    ax.set_xticks(xs[::step])
    ax.set_xticklabels([h.astimezone(tz).strftime("%H%M") for h in s.hours][::step])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", color=S["rule"], lw=0.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0, labelsize=7.8, colors=S["muted"])
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, frameon=False, fontsize=7.8,
              handlelength=1.4)


# --------------------------------------------------------------------------
# Multi-day shift string
# --------------------------------------------------------------------------
def shift_string(grid, summaries, ratings, boundary, cfg, out, issued=None, dpi=150):
    issued = issued or datetime.now(timezone.utc)
    n = len(summaries)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig_h = 4.6 + 5.0 * nrows
    fig = plt.figure(figsize=(16, fig_h), facecolor=S["paper"])
    w0, w1 = summaries[0].window, summaries[-1].window
    overall = max(r.level for r in ratings)
    hh = 1.05 / fig_h
    y = _header(fig, "ZJX ARTCC  ·  SHIFT-STRING WEATHER & WORKLOAD OUTLOOK",
                f"{w0.shift.code} shifts  —  {w0.day:%a %d %b} to {w1.day:%a %d %b %Y}",
                f"{w0.shift.start.replace(':', '')}–{w0.shift.end.replace(':', '')} local each day"
                f"   ·   {n} shifts", None, height=hh)
    _meta_line(fig, y - 0.2 / fig_h, grid, issued)

    # Calendar strip: one column per day, one row per hazard.
    names = ["Overall"] + [h.name for h in ratings[0].hazards]
    top = y - 0.55 / fig_h
    row_h = 0.27 / fig_h
    lab_w = 0.15
    col_w = (0.95 - lab_w) / n
    for j, (s, r) in enumerate(zip(summaries, ratings)):
        cx = 0.025 + lab_w + j * col_w
        fig.text(cx + col_w / 2, top, f"{s.window.day:%a %d %b}", ha="center", va="center",
                 fontsize=10, weight="bold")
        levels = [r.level] + [h.level for h in r.hazards]
        for i, lv in enumerate(levels):
            yy = top - (i + 1) * row_h - 0.06 / fig_h
            c = _cat_color(lv)
            strong = i == 0
            fig.patches.append(FancyBboxPatch(
                (cx + 0.003, yy - row_h * 0.42), col_w - 0.006, row_h * 0.84,
                boxstyle="round,pad=0,rounding_size=0.003", transform=fig.transFigure,
                facecolor=c if strong else _tint(c, 0.22), edgecolor="none"))
            fig.text(cx + col_w / 2, yy, CATEGORY[lv], ha="center", va="center",
                     fontsize=9 if strong else 8, weight="bold",
                     color="white" if strong else c)
    for i, nm in enumerate(names):
        yy = top - (i + 1) * row_h - 0.06 / fig_h
        fig.text(0.025, yy, nm, va="center", fontsize=9.5 if i == 0 else 9,
                 weight="bold" if i == 0 else "normal")

    # Small-multiple composite maps.
    maps_top = top - (len(names) + 1) * row_h - 0.5 / fig_h
    mh = 4.15 / fig_h
    mw, gap = 4.15 / 16 * 1.02, 0.045
    tl = cfg["turbulence"]["levels"]
    lowest = 1.0
    for k, (s, r) in enumerate(zip(summaries, ratings)):
        row, col = divmod(k, ncols)
        x = 0.025 + col * (mw + gap) + (0.95 - ncols * mw - (ncols - 1) * gap) / 2
        yb = maps_top - (row + 1) * mh - row * 0.95 / fig_h
        ax = _map_axes(fig, [x, yb, mw, mh])
        lowest = min(lowest, yb)
        layer_precip_ts(ax, grid, s, compact=True)
        X, Y, D = _fine(grid, s.dti_max)
        ax.contour(X, Y, D, levels=[tl[1]], colors=S["turb"][2], linewidths=1.5, transform=PC, zorder=3)
        ax.contourf(X, Y, D, levels=[tl[1], 1e3], colors=[S["turb"][1]], alpha=0.28, transform=PC, zorder=2.6)
        X, Y, H = _fine(grid, s.ice_hours)
        ax.contourf(X, Y, H, levels=[3, 1e3], colors="none", hatches=["////"], transform=PC, zorder=3)
        ax.contour(X, Y, H, levels=[3], colors=S["ice"], linewidths=1.0, transform=PC, zorder=3)
        _overlay_geography(ax, boundary, label_size=6.5)
        bb = ax.get_position()
        fig.text(bb.x0, bb.y1 + 0.1 / fig_h, f"{s.window.day:%a %d %b}", fontsize=11.5,
                 weight="bold", va="bottom")
        fig.text(bb.x0 + 0.075, bb.y1 + 0.11 / fig_h, s.window.local_label(), fontsize=8,
                 color=S["muted"], va="bottom")
        c = _cat_color(r.level)
        fig.text(bb.x1, bb.y1 + 0.1 / fig_h, f" {r.label} ", ha="right", va="bottom", fontsize=8.5,
                 weight="bold", color="white",
                 bbox=dict(facecolor=c, edgecolor="none", boxstyle="round,pad=0.3"))
        fig.text(bb.x0, bb.y0 - 0.12 / fig_h, _short_reason(r), fontsize=7.6, color=S["muted"], va="top")

    cmap, norm = _precip_cmap()
    handles = [Patch(facecolor=c, label=f"{lo:g}") for c, lo in
               zip(S["precip_colors"], S["precip_levels_in"])]
    handles[0].set_label(f"Precip (in): {S['precip_levels_in'][0]:g}")
    handles += [Line2D([], [], color=S["ts"], lw=1.2, label="TS ≥ 1 h"),
                Patch(facecolor="white", edgecolor=S["ts"], hatch="....", label="TS ≥ 3 h"),
                Patch(facecolor=_tint(S["turb"][1], 0.4), edgecolor=S["turb"][2], lw=1.2,
                      label=f"CAT: DTI ≥ {tl[1]}"),
                Patch(facecolor="white", edgecolor=S["ice"], hatch="////", label="Icing ≥ 3 h")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, lowest - 0.42 / fig_h), ncol=len(handles),
               frameon=False, fontsize=7.8, handlelength=1.3, columnspacing=0.8, handletextpad=0.4)
    _footer(fig)
    fig.savefig(out, dpi=dpi, facecolor=S["paper"])
    plt.close(fig)
    return out


def _short_reason(r):
    if r.level == 2 and all(h.level < 2 for h in r.hazards):
        return r.reason  # compound rule fired
    top = [h for h in r.hazards if h.level == r.level and h.level > 0]
    if not top:
        return "No hazard reaches moderate thresholds"
    return " · ".join(f"{h.name}: {h.driver}" for h in top)
