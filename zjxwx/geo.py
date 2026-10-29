"""Boundary handling, airspace-area statistics and basemap geometry."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from matplotlib.path import Path
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import shape

from .config import DATA_DIR


def load_boundary(path=None):
    """Return the boundary as a shapely (Multi)Polygon."""
    with open(path or DATA_DIR / "kzjx_boundary.geojson") as fh:
        gj = json.load(fh)
    if gj.get("type") == "FeatureCollection":
        gj = gj["features"][0]
    return shape(gj.get("geometry", gj))


def _polygon_path(geom):
    """One matplotlib Path covering every polygon, holes included."""
    verts, codes = [], []
    polys = getattr(geom, "geoms", [geom])
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords)
            verts.append(xy)
            codes.append([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY])
    return Path(np.concatenate(verts), np.concatenate(codes))


class AirspaceMask:
    """Area-weighted statistics of a coarse model field inside a polygon.

    The model grid (0.5-0.75 deg) puts only a few dozen points inside ZJX,
    so a point-in-polygon test on the raw grid makes "% of airspace"
    jumpy and sensitive to which points happen to fall inside. Instead the
    field is bilinearly interpolated to a fine (default 0.05 deg) grid,
    masked by the polygon and weighted by cos(latitude) so each sample
    represents equal area.
    """

    def __init__(self, boundary, lats, lons, res=0.05):
        self.lats, self.lons = lats, lons
        x0, y0, x1, y1 = boundary.bounds
        fy = np.arange(max(y0, lats[0]), min(y1, lats[-1]) + 1e-9, res)
        fx = np.arange(max(x0, lons[0]), min(x1, lons[-1]) + 1e-9, res)
        FX, FY = np.meshgrid(fx, fy)
        inside = _polygon_path(boundary).contains_points(np.c_[FX.ravel(), FY.ravel()])
        self.pts = np.c_[FY.ravel()[inside], FX.ravel()[inside]]
        w = np.cos(np.radians(self.pts[:, 0]))
        self.w = w / w.sum()
        if not len(self.pts):
            raise ValueError("boundary does not overlap the model grid")

    def sample(self, field):
        f = np.asarray(field, dtype=float)
        interp = RegularGridInterpolator((self.lats, self.lons), f, bounds_error=False)
        return interp(self.pts)

    def frac_ge(self, field, thr):
        """Area fraction of the airspace where field >= thr (NaN counts as no)."""
        s = self.sample(field)
        return float(np.sum(self.w * (np.nan_to_num(s, nan=-np.inf) >= thr)))

    def max(self, field):
        s = self.sample(field)
        return float(np.nanmax(s)) if np.isfinite(s).any() else float("nan")

    def frac_series(self, cube, thr):
        """frac_ge for every time step of a (nt, ny, nx) cube."""
        return np.array([self.frac_ge(c, thr) for c in cube])


@lru_cache(maxsize=1)
def basemap(path=None):
    """Shapely geometries by layer from the shipped Natural Earth extract."""
    with open(path or DATA_DIR / "basemap_se.geojson") as fh:
        gj = json.load(fh)
    layers = {}
    for f in gj["features"]:
        layers.setdefault(f["properties"]["layer"], []).append(shape(f["geometry"]))
    return layers
