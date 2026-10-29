# zjx-wx: ZJX pre-shift weather and workload briefings

This package turns GFS output from Open-Meteo into two products for ZJX ARTCC:

* **Briefing sheet** (`brief`): one shift. It has three hazard maps (precipitation and thunderstorms, clear-air turbulence for FL240–FL390, and icing potential with the freezing level), a workload scorecard and an hourly hazard-coverage timeline.
* **Shift-string outlook** (`string`): several days of the same shift. It has a hazard × day calendar strip and a composite map for each day.

It replaces `original/make_gfs_map.py` and `original/make_string_panel.py`. See [CRITIQUE.md](CRITIQUE.md) for the bugs that were fixed and the reasoning behind the science.

## Install
```bash
pip install -r requirements.txt      # numpy scipy matplotlib shapely cartopy
```
The basemap ships in `zjxwx/data/`, so nothing is downloaded at run time.

## Use
```bash
# 1. Fetch GFS (the day before is included automatically for the DTI look-back)
python -m zjxwx fetch  --start 2026-10-30 --days 6 --out gfs.json          # --step 0.25 for finer CAT

# 2. One-shift briefing sheet
python -m zjxwx brief  gfs.json --date 2026-10-30 --shift F --out brief_F.png --json brief_F.json

# 3. Multi-day shift string
python -m zjxwx string gfs.json --start 2026-10-30 --days 6 --shift N --out string_N.png

# Demo without network access
python -m zjxwx synth  --start 2026-10-30 --days 6 --out demo.json
```
Common options:

| Option | What it does |
|---|---|
| `--config my.json` | Overrides any threshold in `zjxwx/config.py` `DEFAULTS` (deep-merged). |
| `--boundary file.geojson` | Uses a different airspace polygon. |
| `--tz` | Sets the time zone for shift times. |
| `--issued 2026-10-29T22:00Z` | Sets the issue time stamp. |
| `--dpi` | Sets the image resolution. |
| `--out *.pdf` / `*.svg` | Writes vector output. |

Shift times live in `config.SHIFTS` (F 0515–1315 and N 1300–2100 local). Edit them there or add more shifts.

Grid JSON from the original scripts (`{"pts": [...], "data": [...]}`) still loads, as long as it contains the 400–200 hPa wind and height fields. Fetch new data with `fetch` to get the full set of fields.

## Layout
```
zjxwx/config.py       domain, shifts, thresholds, palette, disclaimer
zjxwx/data.py         grid JSON -> (time, lat, lon) arrays, unit handling
zjxwx/diagnostics.py  spherical kinematics, Ellrod TI1 / Ellrod-Knox DTI, icing potential, FZRA, TS hours
zjxwx/analysis.py     shift windows (DST-aware), per-shift summaries, workload rating
zjxwx/geo.py          boundary, area-weighted airspace statistics, basemap
zjxwx/render.py       briefing sheet and shift-string graphics
zjxwx/fetch.py        Open-Meteo GFS client (batched, retried)
zjxwx/synthetic.py    synthetic demo scenario
tools/build_basemap.py  rebuilds the bundled Natural Earth extract
tests/                physics and time-window checks (python -m pytest tests)
samples/              example output (synthetic data) and the original script's output for comparison
```

## Automated runs (GitHub Actions)
`.github/workflows/briefings.yml` runs at 05:40Z and 17:40Z, after Open-Meteo has the 00Z and 12Z GFS cycles. You can also start it by hand from the **Actions** tab ("Run workflow"). Each run does three things:

1. Fetches six days of GFS (about 3,100 Open-Meteo weighted calls).
2. Renders the next F and N briefing sheets and the 5-day F and N shift strings.
3. Force-pushes the images, plus a `README.md` summary, to the **`briefings`** branch. They're also kept as a run artifact for 14 days.

**Open-Meteo terms:** the free API is for non-commercial use and allows 10,000 weighted calls per day. Two scheduled runs and one manual run fit within that. For customer distribution, buy an Open-Meteo API subscription and add the key as the repository secret `OPEN_METEO_API_KEY`. The fetcher then switches to the customer endpoint automatically.
