"""One panel image per shift string.
Usage: make_string_panel.py <grid_json> <start_date> <days> <shift F|N> <out.png>
Each panel: precip sum (in, blue), 200 mb streamlines, Ellrod contours
(orange), icing-potential hours (blue hatch), VATSIM KZJX boundary.
Workload (from in-boundary GFS stats):
  HIGH if Ellrod max >= 8, or icing>=6h over >=50% of boundary,
        or precip >1.0in over >=10% of boundary.
  MEDIUM if Ellrod max >= 4, or icing>=3h over >=25%,
        or precip >0.5in over >=10%.
  Else LOW.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from datetime import date, timedelta

grid_json, start, ndays, shift, out = sys.argv[1:6]
shift = shift.upper(); ndays = int(ndays)
if shift == "F":
    hours = list(range(6, 18)); snap = "12:00"; win_label = "06Z–18Z"; snap_label = "12Z"; shift_label = "F 0515–1315"
else:
    hours = list(range(12, 24)); snap = "18:00"; win_label = "12Z–00Z"; snap_label = "18Z"; shift_label = "N 1300–2100"
gj = json.load(open(grid_json))
pts = gj["pts"]; data = gj["data"]
lats = sorted(set(p[0] for p in pts)); lons = sorted(set(p[1] for p in pts))
nl, no = len(lats), len(lons)
kz = json.load(open("/home/hatch/workspace/goals/zjx-pre-shift-weather-briefings/data/kzjx_boundary.geojson"))
ring = np.array(kz["geometry"]["coordinates"][0][0])
inside = np.array([Path(ring).contains_point((lo, la)) for (la, lo) in pts])
states = json.load(open("/home/hatch/workspace/goals/zjx-pre-shift-weather-briefings/data/ne_states.geojson"))
def uv(s, d):
    s = (s or 0) / 3.6; r = np.radians(d or 0); return -s * np.sin(r), -s * np.cos(r)
y0, m0, d0 = map(int, start.split("-"))
days = [date(y0, m0, d0) + timedelta(days=i) for i in range(ndays)]
ncols = 3; nrows = (ndays + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5.2 * nrows), squeeze=False)
fig.suptitle(f"ZJX shift string — {days[0].strftime('%a %b %d')} to {days[-1].strftime('%a %b %d, %Y')}  (all {shift} shifts)",
             fontsize=17, weight="bold")
levels = [0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
for ax, day in zip(axes.flat, days):
    ds = day.strftime("%Y-%m-%d")
    precip = np.zeros((nl, no)); ice = np.zeros((nl, no))
    U300 = np.full((nl, no), np.nan); V300 = np.full_like(U300, np.nan)
    U500 = np.full_like(U300, np.nan); V500 = np.full_like(U300, np.nan)
    Z300 = np.full_like(U300, np.nan); Z500 = np.full_like(U300, np.nan)
    U200 = np.full_like(U300, np.nan); V200 = np.full_like(U300, np.nan)
    for (la, lo), loc in zip(pts, data):
        h = loc["hourly"]; t = h["time"]; r = lats.index(la); c = lons.index(lo)
        for hh in hours:
            i = t.index(f"{ds}T{hh:02d}:00")
            precip[r, c] += h["precipitation"][i] or 0.0
            for lev in ("850", "700", "500"):
                temp = h[f"temperature_{lev}hPa"][i]; rh = h[f"relative_humidity_{lev}hPa"][i]
                if temp is not None and rh is not None and -18 <= temp <= 0 and rh >= 75:
                    ice[r, c] += 1; break
        i = t.index(f"{ds}T{snap}")
        U300[r, c], V300[r, c] = uv(h["wind_speed_300hPa"][i], h["wind_direction_300hPa"][i])
        U500[r, c], V500[r, c] = uv(h["wind_speed_500hPa"][i], h["wind_direction_500hPa"][i])
        Z300[r, c] = h["geopotential_height_300hPa"][i]; Z500[r, c] = h["geopotential_height_500hPa"][i]
        U200[r, c], V200[r, c] = uv(h["wind_speed_200hPa"][i], h["wind_direction_200hPa"][i])
    precip = precip / 25.4
    dUdy, dUdx = np.gradient(U300, 0.75, 0.75); dVdy, dVdx = np.gradient(V300, 0.75, 0.75)
    coslat = np.cos(np.radians(lats))[:, None]
    dUdx = dUdx / (111320 * coslat); dUdy = dUdy / 111320
    dVdx = dVdx / (111320 * coslat); dVdy = dVdy / 111320
    DEF = np.sqrt((dUdx - dVdy) ** 2 + (dVdx + dVdy) ** 2)
    TI = DEF * np.hypot(U300 - U500, V300 - V500) / np.maximum(Z300 - Z500, 500.0) * 1e7
    Lon, Lat = np.meshgrid(lons, lats)
    ax.contourf(Lon, Lat, precip, levels=levels, cmap="Greens", extend="max", alpha=0.85)
    for feat in states["features"]:
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            for ring2 in poly:
                ax.plot([p[0] for p in ring2], [p[1] for p in ring2], color="#444444", lw=0.5)
    ax.plot(ring[:, 0], ring[:, 1], color="#b00000", lw=1.6, linestyle="--")
    cs = ax.contourf(Lon, Lat, ice, levels=[2.5, 5.5, 12.5], colors="none", hatches=["/", "//"])
    for col in cs.collections:
        col.set_edgecolor("#0066cc"); col.set_linewidth(0.5)
    ax.contour(Lon, Lat, TI, levels=[2, 4, 8], colors="#e67300", linewidths=[0.8, 1.2, 1.8])
    ax.streamplot(Lon, Lat, U200, V200, density=0.7, color="#555555", linewidth=0.6, arrowsize=0.7)
    for name, (x, y) in {"JAX": (-81.66, 30.33), "SAV": (-81.20, 32.13), "TLH": (-84.35, 30.39),
                         "TPA": (-82.53, 27.98), "ORL": (-81.38, 28.55)}.items():
        ax.plot(x, y, "ko", ms=3)
        ax.text(x + 0.12, y + 0.1, name, fontsize=7, weight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5))
    pi = precip.ravel()[inside]; ti = TI.ravel()[inside]; ic = ice.ravel()[inside]
    ti_max = np.nanmax(ti); pmax = pi.max(); imax = ic.max()
    if ti_max >= 8 or np.mean(ic >= 6) >= 0.50 or np.mean(pi > 1.0) >= 0.10:
        wl, wcol = "HIGH", "#c00000"
    elif ti_max >= 4 or np.mean(ic >= 3) >= 0.25 or np.mean(pi > 0.5) >= 0.10:
        wl, wcol = "MEDIUM", "#d88000"
    else:
        wl, wcol = "LOW", "#2e7d32"
    ax.set_title(f"{day.strftime('%a %b %d')} — {shift_label}", fontsize=12, weight="bold", loc="left")
    ax.text(0.02, 0.96, f"Workload: {wl}", transform=ax.transAxes, fontsize=11, weight="bold",
            color="white", va="top",
            bbox=dict(facecolor=wcol, edgecolor="none", pad=4))
    ax.text(0.02, 0.02, f"Rain max {pmax:.2f}″ · Ellrod max {ti_max:.0f} · Icing max {imax:.0f}h",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
    ax.set_xlim(-88.5, -76.0); ax.set_ylim(23.5, 35.0)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes.flat[ndays:]:
    ax.axis("off")
fig.text(0.5, 0.01,
         f"GFS via Open-Meteo · Precip sum {win_label} (in, greens) · 200 mb streamlines {snap_label}Z · "
         "Ellrod contours 2/4/8 (orange, x1e-7 s^-2) · Icing-potential hours: / = 3–5h, // = 6+h (blue) · "
         "Wind barbs omitted at panel scale · Workload from in-boundary stats: HIGH if Ellrod max ≥8, icing 6+h over ≥50%, or rain >1″ over ≥10%; "
         "MEDIUM if Ellrod max ≥4, icing 3+h over ≥25%, or rain >0.5″ over ≥10%; else LOW · Boundary: VATSIM KZJX polygon (not FAA-official)",
         ha="center", fontsize=9, style="italic", wrap=True)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(out, dpi=150)
print("wrote", out)
