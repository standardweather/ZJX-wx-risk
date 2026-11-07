# zjx-wx: ZJX pre-shift weather and workload briefings

This package turns GFS output from Open-Meteo into two products for ZJX ARTCC:

* **Briefing sheet** (`brief`): one shift. It has three hazard maps (precipitation and thunderstorms, clear-air turbulence for FL240–FL390, and icing potential with the freezing level), a workload scorecard and an hourly hazard-coverage timeline.
* **Shift-string outlook** (`string`): several days of the same shift. It has a hazard × day calendar strip and a composite map for each day.

* **OM/TMU/OS maps** (`nbm` + `tmu`): one simple ZJX map for each of 0500–1300, 1300–2100 and 2100–0500 local, issued once each morning. Each shows **hazard** areas (thunderstorms, severe turbulence, severe icing) and **awareness** areas (IFR/LIFR, moderate turbulence, moderate icing, low-level wind shear). See [OM/TMU/OS maps](#omtmuos-maps).

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

## OM/TMU/OS maps
```bash
python -m zjxwx nbm  --date today --out nbm.npz     # ~25 s, ~150 MB of range requests
python -m zjxwx hrrr --date today --out hrrr.npz    # ~5 s, echo tops + lightning
python -m zjxwx tmu gfs.json --nbm nbm.npz --hrrr hrrr.npz --date today --outdir tmu  # tmu_AM/PM/MID.png, tmu.json
```
The National Blend of Models is free on NOAA's AWS open-data bucket (`noaa-nbm-grib2-pds`), with no key or quota. Only the needed GRIB2 messages are downloaded, using the `.idx` byte offsets. NBM is used wherever it has the field. GFS covers what NBM lacks:

| Area | Type | Source and rule (any hour of the period, thresholds in `config.DEFAULTS["tmu"]`) |
|---|---|---|
| Thunderstorms | Hazard | Any of: HRRR echo top ≥ 30 kft or lightning (widened 20 km for placement error); NBM 1-h thunder probability ≥ 25 %; GFS thunderstorm signal (CAPE, LI, precip) |
| Severe turbulence | Hazard | GFS Ellrod–Knox DTI ≥ 12, FL240–FL390 |
| Severe icing | Hazard | NBM freezing rain (1-h ice accretion ≥ 0.1 mm), or GFS icing potential ≥ 0.9 on ≥ 3 levels at once (a deep saturated supercooled layer) |
| IFR / LIFR | Awareness | NBM P(ceiling < 1000 ft or vis < 3 SM) ≥ 40 %; LIFR cross-hatched where P(ceiling < 500 ft or vis < 1 SM) ≥ 40 %. Land only: NBM leaves these undefined over water |
| Moderate turbulence | Awareness | GFS DTI ≥ 8, FL240–FL390 |
| Moderate icing | Awareness | GFS icing potential ≥ 0.5 for ≥ 2 h, FL025–FL210 |
| Low-level wind shear | Awareness | NBM LLWS diagnosed below 2,000 ft AGL |

Why HRRR for thunderstorms: on 6 Nov 2026, storms with 42–49 kft MRMS echo tops crossed South Carolina from 10Z. NBM 05Z gave 0 % thunder there, but HRRR 00Z and 04Z both forecast 35–42 kft tops with lightning in the right place. HRRR (also on AWS open data, `noaa-hrrr-bdp-pds`) is a 3 km convection-allowing model. Each valid hour uses the newest cycle that reaches it: hourly cycles go to 18 h, and 00/06/12/18Z go to 48 h.

Why GFS for turbulence and icing: NBM v4 has no in-flight icing aloft. Its only upper-level turbulence field is an undocumented 0–3 Ellrod category for 400–300 hPa. Its low-level turbulence category (surface–4,500 ft) put "severe" and "extreme" over 6 % of ZJX on 6 Nov 2026, where 10 m winds were about 8 kt, so it is not used. Without an NBM file (or if the download fails), `tmu` still renders: thunderstorms and freezing rain fall back to GFS, and IFR/LIFR and LLWS show as unavailable.

The side panel keeps the words to a minimum: each item's share of ZJX, local hours and altitudes or tops, plus the ZJX airports that reach P(IFR) ≥ 40 %. The footer reads only "Unofficial Briefing Aid - Automatically Generated"; the model runs used are recorded in `tmu.json`.

Areas are generalised for display (`close_km`, `min_area_km2`, `smooth_km` in `config.DEFAULTS["tmu"]`): gaps under 20 km are closed, areas under 5,000 km² and holes of that size are dropped, and edges are smoothed over 15 km. The share of ZJX is measured on what is drawn.

## Layout
```
zjxwx/config.py       domain, shifts, thresholds, palette, disclaimer
zjxwx/data.py         grid JSON -> (time, lat, lon) arrays, unit handling
zjxwx/diagnostics.py  spherical kinematics, Ellrod TI1 / Ellrod-Knox DTI, icing potential, FZRA, TS hours
zjxwx/analysis.py     shift windows (DST-aware), per-shift summaries, workload rating
zjxwx/geo.py          boundary, area-weighted airspace statistics, basemap
zjxwx/render.py       briefing sheet and shift-string graphics
zjxwx/fetch.py        Open-Meteo GFS client (batched, retried)
zjxwx/hrrr.py         HRRR echo tops and lightning from AWS open data (newest cycle per hour)
zjxwx/nbm.py          NBM GRIB2 download from AWS open data (byte ranges), cut to the map extent
zjxwx/tmu.py          OM/TMU/OS hazard and awareness maps
zjxwx/synthetic.py    synthetic demo scenario
tools/build_basemap.py  rebuilds the bundled Natural Earth extract
tests/                physics and time-window checks (python -m pytest tests)
samples/              example output (synthetic data) and the original script's output for comparison
```

## Automated runs (GitHub Actions)
`.github/workflows/briefings.yml` runs in the morning (06:15Z, 07:30Z or 08:45Z, whichever GitHub starts first) and at 17:40Z, after Open-Meteo has the 00Z and 12Z GFS cycles. You can also start it by hand from the **Actions** tab ("Run workflow"). Each run does three things:

1. Fetches six days of GFS (about 3,100 Open-Meteo weighted calls).
2. Renders the next F and N briefing sheets and the 5-day F and N shift strings.
   The OM/TMU/OS maps are made once a day. The morning is scheduled three times (06:15Z, 07:30Z and 08:45Z, i.e. 01:15–03:45 EST), because GitHub starts scheduled runs late and sometimes drops them. The first run to start makes the maps and the others exit at once. The 17:40Z run carries the morning's maps forward, or makes them if no morning run happened. Manual runs always make fresh maps.

   **GitHub's scheduler is best-effort.** In Nov 2026 scheduled runs here started 4–8 hours late, and on 7 Nov the morning run never started. For a hard 0500L deadline, trigger the workflow from an external cron service (see below).
3. Force-pushes the images, plus a `README.md` summary, to the **`briefings`** branch. They're also kept as a run artifact for 14 days.

**Open-Meteo terms:** the free API is for non-commercial use and allows 10,000 weighted calls per day. Two scheduled runs and one manual run fit within that. For customer distribution, buy an Open-Meteo API subscription and add the key as the repository secret `OPEN_METEO_API_KEY`. The fetcher then switches to the customer endpoint automatically.
