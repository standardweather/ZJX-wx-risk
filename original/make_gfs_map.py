"""GFS string map for ZJX domain (one day).
Usage: make_gfs_map.py <grid_json> <date> <shift F|N> <out.png>
  F shift: precip summed 06Z-17Z (12 hourly values), winds/Ellrod at 12Z.
  N shift: precip summed 12Z-23Z (12 hourly values), winds/Ellrod at 18Z.
Grid JSON: {"pts":[[lat,lon],...], "data":[Open-Meteo hourly,...]}
Layers: precip sum (blue fill, inches), 200 mb barbs + streamlines,
  Ellrod turbulence index contours (300 hPa deformation x 300-500 hPa
  shear, x1e-7 s^-2; outlines at 2/4/8), icing-potential hours hatched
  (hours in window with T in [-18,0]C and RH>=75% at 850/700/500 hPa).
ZJX boundary: VATSIM vatspy-data-project KZJX polygon (53 vertices,
data/kzjx_boundary.geojson) — VATSIM mirror, NOT FAA-official.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

grid_json, date, shift, out = sys.argv[1:5]
shift = shift.upper()
if shift == "F":
    hours = [f"{h:02d}:00" for h in range(6, 18)]; wind_hour = "12:00"; win_label = "06Z–18Z"; snap = "12Z"
else:
    hours = [f"{h:02d}:00" for h in range(12, 24)]; wind_hour = "18:00"; win_label = "12Z–00Z"; snap = "18Z"
gj = json.load(open(grid_json))
pts = gj["pts"]; data = gj["data"]
lats = sorted(set(p[0] for p in pts)); lons = sorted(set(p[1] for p in pts))
nl, no = len(lats), len(lons)
precip = np.full((nl, no), np.nan); wspd = np.full((nl, no), np.nan); wdir = np.full((nl, no), np.nan)
ice_hrs = np.zeros((nl, no))
U300 = np.full((nl, no), np.nan); V300 = np.full_like(U300, np.nan)
U500 = np.full_like(U300, np.nan); V500 = np.full_like(U300, np.nan)
Z300 = np.full_like(U300, np.nan); Z500 = np.full_like(U300, np.nan)
def uv(spd_kmh, direc):
    s = (spd_kmh or 0) / 3.6; r = np.radians(direc or 0)
    return -s * np.sin(r), -s * np.cos(r)
for (la, lo), loc in zip(pts, data):
    h = loc["hourly"]; t = h["time"]
    r = lats.index(la); c = lons.index(lo)
    total = 0.0; ok = False
    for hh in hours:
        key = f"{date}T{hh}"
        if key in t:
            i = t.index(key)
            total += h["precipitation"][i] or 0.0; ok = True
            icing = False
            for lev in ("850", "700", "500"):
                temp = h[f"temperature_{lev}hPa"][i]; rh = h[f"relative_humidity_{lev}hPa"][i]
                if temp is not None and rh is not None and -18 <= temp <= 0 and rh >= 75:
                    icing = True
            if icing: ice_hrs[r, c] += 1
    if ok: precip[r, c] = total / 25.4
    wkey = f"{date}T{wind_hour}"
    if wkey in t:
        i = t.index(wkey)
        wspd[r, c] = (h["wind_speed_200hPa"][i] or 0) * 0.539957
        wdir[r, c] = h["wind_direction_200hPa"][i] or 0
        U300[r, c], V300[r, c] = uv(h["wind_speed_300hPa"][i], h["wind_direction_300hPa"][i])
        U500[r, c], V500[r, c] = uv(h["wind_speed_500hPa"][i], h["wind_direction_500hPa"][i])
        Z300[r, c] = h["geopotential_height_300hPa"][i]
        Z500[r, c] = h["geopotential_height_500hPa"][i]
Lon, Lat = np.meshgrid(lons, lats)
rad = np.radians(wdir)
U = -wspd * np.sin(rad); V = -wspd * np.cos(rad)
# Ellrod index: 300 hPa total deformation x 300-500 hPa vector shear
dUdy, dUdx = np.gradient(U300, 0.75, 0.75)
dVdy, dVdx = np.gradient(V300, 0.75, 0.75)
m_per_deg = 111320.0; coslat = np.cos(np.radians(lats))[:, None]
dUdx = dUdx / (m_per_deg * coslat); dUdy = dUdy / m_per_deg
dVdx = dVdx / (m_per_deg * coslat); dVdy = dVdy / m_per_deg
DEF = np.sqrt((dUdx - dVdy) ** 2 + (dVdx + dVdy) ** 2)
shear = np.hypot(U300 - U500, V300 - V500) / np.maximum(Z300 - Z500, 500.0)
TI = DEF * shear * 1e7

fig, ax = plt.subplots(figsize=(12, 9))
levels = [0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
cf = ax.contourf(Lon, Lat, precip, levels=levels, cmap="Greens", extend="max", alpha=0.85)
cb = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
cb.set_label(f"GFS precipitation sum {win_label} (in)")
gj_states = json.load(open("/home/hatch/workspace/goals/zjx-pre-shift-weather-briefings/data/ne_states.geojson"))
for feat in gj_states["features"]:
    geom = feat["geometry"]
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    for poly in polys:
        for ring in poly:
            ax.plot([p[0] for p in ring], [p[1] for p in ring], color="#444444", lw=0.6)
kzjx_geo = json.load(open("/home/hatch/workspace/goals/zjx-pre-shift-weather-briefings/data/kzjx_boundary.geojson"))
for poly in kzjx_geo["geometry"]["coordinates"]:
    ring = poly[0]
    ax.plot([p[0] for p in ring], [p[1] for p in ring], color="#b00000", lw=2.2, linestyle="--")
ax.text(-87.6, 34.6, "ZJX boundary (VATSIM vatspy-data-project, KZJX feature)", color="#b00000", fontsize=10, weight="bold")
# Icing-potential hours: hatched, blue (AWC-style)
cs = ax.contourf(Lon, Lat, ice_hrs, levels=[2.5, 5.5, 12.5], colors="none",
                 hatches=["/", "//"])
for col in cs.collections:
    col.set_edgecolor("#0066cc"); col.set_linewidth(0.6)
# Ellrod contours: orange outlines (AWC-style)
ec = ax.contour(Lon, Lat, TI, levels=[2, 4, 8], colors="#e67300", linewidths=[1.0, 1.6, 2.2])
ax.clabel(ec, fmt={2: "Ellrod 2", 4: "Ellrod 4", 8: "Ellrod 8"}, fontsize=9, colors="#e67300")
ax.barbs(Lon[::2, ::2], Lat[::2, ::2], U[::2, ::2], V[::2, ::2], length=5.5, color="#222222")
ax.streamplot(Lon, Lat, U, V, density=0.7, color="#555555", linewidth=0.7, arrowsize=0.8)
ax.text(-87.6, 23.9, "Barbs + streamlines: 200 mb winds at snapshot hour (kt)", fontsize=10, style="italic")
cities = {"Jacksonville":(-81.66,30.33),"Savannah":(-81.20,32.13),"Tallahassee":(-84.35,30.39),
          "Tampa":(-82.53,27.98),"Orlando":(-81.38,28.55),"Miami":(-80.29,25.79)}
for name,(x,y) in cities.items():
    ax.plot(x, y, "ko", ms=4); ax.text(x+0.15, y+0.12, name, fontsize=10, weight="bold",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))
ax.legend(handles=[Patch(facecolor="white", edgecolor="#0066cc", hatch="/", label="Icing potential 3–5 hrs"),
                   Patch(facecolor="white", edgecolor="#0066cc", hatch="//", label="Icing potential 6+ hrs"),
                   Patch(facecolor="white", edgecolor="#e67300", label="Ellrod contours 2 / 4 / 8 (x1e-7 s^-2)")],
          loc="lower right", fontsize=9, framealpha=0.9)
ax.set_xlim(-88.5, -76.0); ax.set_ylim(23.5, 35.0)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title(f"GFS precip sum {win_label} + 200 mb winds ({snap}Z) — {date} ({shift} shift)",
             fontsize=14, weight="bold")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(out, dpi=150)
print("wrote", out)
