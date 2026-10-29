"""Build the offline basemap shipped in zjxwx/data/basemap_se.geojson.

Clips Natural Earth land, lakes, admin-0 and admin-1 boundaries to the
southeast US so the briefing renders with no network access at run time.

Usage: python tools/build_basemap.py <natural_earth_dir> [out.geojson]

<natural_earth_dir> must hold the unzipped shapefile folders
ne_10m_land, ne_50m_lakes, ne_50m_admin_0_countries and
ne_10m_admin_1_states_provinces_lakes (https://www.naturalearthdata.com).
"""
import json
import sys
from pathlib import Path

import shapely
from shapely.geometry import box, mapping, shape
from cartopy.io import shapereader

BBOX = (-93.0, 20.5, -71.0, 37.5)  # lon0, lat0, lon1, lat1 (margin around the map)
TOL = 0.01  # deg simplification, ~1 km


def _features(path):
    reader = shapereader.Reader(str(path))
    for rec in reader.records():
        yield rec.geometry, rec.attributes


def main(ne_dir, out):
    ne = Path(ne_dir)
    clip = box(*BBOX)
    feats = []

    def add(geom, layer, **props):
        g = geom.intersection(clip)
        if g.is_empty:
            return
        g = shapely.simplify(g, TOL, preserve_topology=True)
        feats.append({"type": "Feature", "properties": {"layer": layer, **props},
                      "geometry": mapping(g)})

    land = shapely.union_all([g.intersection(clip) for g, _ in
                              _features(ne / "ne_10m_land/ne_10m_land.shp") if g.intersects(clip)])
    add(land, "land")
    for g, a in _features(ne / "ne_50m_lakes/ne_50m_lakes.shp"):
        if g.intersects(clip):
            add(g, "lake", name=a.get("name"))
    for g, a in _features(ne / "ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp"):
        if g.intersects(clip):
            add(g.boundary, "country", name=a.get("NAME"))
    shp = ne / "ne_10m_admin_1_states_provinces_lakes/ne_10m_admin_1_states_provinces_lakes.shp"
    for g, a in _features(shp):
        if a.get("adm0_a3") == "USA" and g.intersects(clip):
            add(g.boundary, "state",
                name=a.get("name"), abbrev=a.get("postal"))

    Path(out).write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                                    separators=(",", ":")))
    print(f"wrote {out} ({len(feats)} features)")


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[1]
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else here / "zjxwx/data/basemap_se.geojson")
