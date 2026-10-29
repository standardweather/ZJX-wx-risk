# ZJX workload forecast: review of the original scripts and what changed

This covers `make_gfs_map.py` and `make_string_panel.py` (kept unchanged in `original/`) and the `zjxwx` package that replaces them.

## 1. Bugs in the original

| # | Problem | Effect | Fix |
|---|---|---|---|
| 1 | The shearing-deformation term used `dVdx + dVdy` instead of `dVdx + dUdy`. | Total deformation, and so the Ellrod index, was wrong everywhere. A pure stretching flow showed up as shear, and real shear along the jet axis (`dU/dy`) was ignored. | `diagnostics.kinematics`, checked by `tests/test_original_shearing_term_bug_is_fixed`. |
| 2 | Derivatives ignored spherical curvature (`u·tanφ/a`, `v·tanφ/a`). | At 30°N in a 100 kt jet these terms are about 4×10⁻⁶ s⁻¹, the same size as the deformation itself. Solid-body rotation showed false deformation. | The curvature terms are now included and tested against solid-body rotation. |
| 3 | `uv()` turned missing wind (`None`) into calm (`0`). | A single missing value created a large fake gradient, which could produce a bogus Ellrod maximum. | Missing values stay NaN all the way through. |
| 4 | Thickness was floored with `np.maximum(Z300 - Z500, 500.0)`. | The floor never triggers (300–500 hPa is about 3.6 km), so it does nothing. If it ever did, it would hide bad data. | Layers with zero or negative thickness are masked. |
| 5 | Grid spacing was hard-coded (`np.gradient(..., 0.75, 0.75)`). | Any other grid gives silently wrong derivatives. | Spacing now comes from the actual coordinates. |
| 6 | Precip windows were off by one hour. | Open-Meteo `precipitation` at hour T is the total for the hour *ending* at T. Summing the 06…17 values covers 05Z–17Z, but the label says 06Z–18Z. | `Window.accum_hours()` picks the hour-ending totals that overlap the shift. |
| 7 | Shift windows didn't match the shifts. | The F shift (0515–1315 local) is 0915–1715Z under EDT and 1015–1815Z under EST, but the code used fixed 06–18Z. The N shift (1300–2100 local) used 12–00Z, which dropped the last 1–2 hours. | Windows are built from local shift times with `zoneinfo`, so DST is handled. |
| 8 | `f"{snap_label}Z"` with `snap_label="12Z"` | Prints "12ZZ" (visible in `samples/before_original_map_*.png`). | Fixed. |
| 9 | `cs.collections` was removed in Matplotlib 3.10. | **The original scripts crash on current Matplotlib.** | `ContourSet` is styled directly. |
| 10 | Absolute `/home/hatch/...` paths | Runs on only one machine. | Bundled data with `--boundary` and `--config` overrides. |
| 11 | `lats.index()` and `t.index()` were called inside nested loops. | O(n²), and it throws `ValueError` when an hour is missing. | One vectorised load into (time, lat, lon) arrays. Missing hours are reported, not fatal. |
| 12 | Point-in-polygon used only `coordinates[0][0]`. | A multi-part boundary or one with holes would be wrong. | Every polygon and hole is handled. |
| 13 | "% of boundary" was the share of raw 0.75° grid points inside the polygon (only about 25 points). | Coverage jumped in steps of about 4%, and edge points counted fully. | `AirspaceMask`: bilinear interpolation to 0.05°, then a cos(lat) area-weighted fraction. |
| 14 | The docstring said precip was blue, but the colormap was `Greens`. 200 mb barbs were plotted with 300 mb Ellrod and nothing said so. | Confusing for readers. | Every panel says what it shows, at which level and when. |

## 2. Science changes

### Clear-air turbulence: Ellrod TI1 becomes Ellrod–Knox DTI over FL240–FL390
* **Original:** one layer (300–500 hPa shear × 300 hPa deformation) at one snapshot hour. 500 hPa is about FL180, which is below where most CAT that matters for en-route traffic occurs, and one snapshot misses anything that develops during the shift.
* **Now:** TI1 = VWS × DEF (Ellrod & Knapp 1992) is computed for four stacked layers, 400–350, 350–300, 300–250 and 250–200 hPa (about FL240–FL390). VWS is |ΔV|/Δz over each layer and DEF uses the layer-mean wind. The product shows the **column maximum at every hour of the shift** and the flight level of the layer that holds it.
* **Divergence trend (Ellrod & Knox 2010):** DTI = TI + DVT, where DVT = C·(DIV(t) − DIV(t−6 h)) and C = 0.1. The authors found this term reduced false alarms and caught CAT in anticyclonic, upper-ridge flow that TI1 alone missed. The fetch step pulls the previous day so the 6-hour look-back exists at the start of a shift. *Caveat:* the paper calls C "subjectively assigned" and does not state the units of the divergence term. Here divergence is expressed in 10⁻⁷ s⁻¹ so that DVT is the same size as TI when divergence changes are large, which is the scaling the paper describes. **This should be verified against PIREPs before anyone relies on it.**
* **First real-data run (29 Oct 2026, 12Z GFS):** two changes came out of checking the output against the raw data. (1) With divergence in 10⁻⁷ s⁻¹ the DVT term had a median of about 12, against about 1 for TI, and it flagged all of ZJX as HIGH every day. Divergence is now expressed in 10⁻⁶ s⁻¹, which gives DVT a median near 1 and a 99th percentile near 10, matching the paper's intent. (2) Inside a convective Gulf low, upper-level outflow produced 180° wind reversals over 1 km and DTI values of 40–100. Both terms are clear-air diagnostics, so the index is now zeroed wherever the model has ≥ 2 mm/h of precipitation nearby in the past 3 hours. That turbulence is counted under thunderstorms instead. TI and divergence are also smoothed over 3×3 points, because hourly 0.5° GFS divergence is noisy.
* **Thresholds:** 4 / 8 / 12 (light-moderate / moderate / moderate-severe), following AWC practice and the paper's finding that 4 best separates turbulent from smooth. VWS in kt per 1000 ft is also computed (the ≥6 kt/1000 ft rule of thumb) for future use.
* **Grid:** the default fetch is now 0.5° instead of 0.75°. Deformation is a derivative, so a coarse grid damps it. `--step 0.25` is better again but uses about 4× the API calls.

### Icing: a binary rule becomes a CIP/FIP-style potential
* **Original:** a level counted as icing if −18 ≤ T ≤ 0 °C and RH ≥ 75 %, checked only at 850, 700 and 500 hPa. A dry −17 °C layer counted the same as a saturated −6 °C cloud, everything between those three levels was skipped, and 0 °C (where supercooled liquid water is rare) counted fully.
* **Now:** at 10 levels from 925 to 450 hPa (about 2,500 to 21,000 ft), *potential = temperature interest × moisture interest*:
  * The temperature interest is 0 above 0 °C, 1 from −3 to −12 °C (where supercooled liquid is most frequent; Bernstein et al. 2005 CIP), 0.3 at −20 °C and 0 at −25 °C (glaciation).
  * The moisture interest ramps from RH 70 % to 95 %. GFS reports RH with respect to liquid water, which is the right reference for supercooled-liquid clouds.
  * An "icing hour" is a column maximum ≥ 0.5. The scorecard reports the affected layer (for example FL080–FL140) and the median freezing level inside ZJX.
* **Freezing rain/drizzle (SLD) flag:** surface T ≤ 0 °C, precipitation ≥ 0.25 mm/h and a warm nose above +1 °C at 925, 850 or 700 hPa. This is rare in ZJX but high impact for SE Georgia and the Carolinas in winter, so any coverage of 2 % or more raises icing to HIGH.
* **Not yet included:** GFS cloud-water, supercooled-liquid and vertical-velocity terms (the main remaining gaps compared with real FIP). Open-Meteo does not serve cloud liquid water.

### Convection: new, and the biggest workload driver ZJX was missing
The original had no convective diagnostic. In Florida, thunderstorms are what drive reroutes, deviations and sector complexity for most of the year. A **thunderstorm hour** now requires all of:
* CAPE ≥ 500 J/kg (instability),
* lifted index ≤ −2 (a second check on instability),
* CIN no stronger than −100 J/kg (the cap can break), and
* model precipitation ≥ 1 mm/h (the model is actually producing convection, not just unstable).

Maps show TS ≥ 1 h (outline) and ≥ 3 h (stippled). The scorecard reports coverage, the maximum number of hours and the maximum CAPE. All thresholds are in `config.py`.

### Workload rating
* Each hazard is rated LOW, MODERATE or HIGH from the **area-weighted fraction of ZJX airspace** that meets a threshold, not from a single grid-point maximum, which is noisy and can sit on the boundary edge.
* **Compounding:** two or more MODERATE hazards together give HIGH overall. Moderate convection plus moderate CAT is a heavier shift than either one alone.
* The rule that fired is printed on the sheet ("Turbulence (CAT): DTI ≥8 over 25% of airspace"), so the rating can be audited. It is also written to JSON with `--json` for logging and later verification.
* **Calibration:** the thresholds are reasoned starting points, not tuned values. The best next step is to log `--json` output every day and compare it with actual traffic counts, delays and reroutes.

## 3. Visual design
* **Projection:** Lambert Conformal centred on ZJX replaces raw lat/lon degrees, which stretch east–west distances by about 15 % at 30°N. The map uses a Natural Earth basemap bundled for offline use (land, coast, lakes, states and countries, including Cuba and the Bahamas, which the old states file lacked).
* **Focus:** areas outside ZJX are faded, and the boundary has a white outline so it reads against any fill.
* **One hazard per panel** (precip and TS, CAT, icing) replaces one map with five overlapping layers. Colours follow AWC conventions: convection red, turbulence orange and brown, icing blue, precipitation WPC-style greens. Every layer has its own legend or colour bar.
* **Branded header:** product name, shift and date, local and UTC windows, a large workload badge, then model, run, grid and issue time. The footer carries the disclaimer.
* **Scorecard and timeline:** each hazard's rating with its key numbers, plus hourly % of airspace affected through the shift, so a supervisor can see *when* in the shift the load peaks.
* **Shift-string outlook:** a calendar strip (days × hazards) above small composite maps, each with a badge and the reason for its rating.
* Fonts use Inter (falling back to DejaVu Sans), and output works as PNG, PDF or SVG by changing the `--out` extension.

## 4. Suggested next steps
1. **Verify** DTI against PIREPs and EDR reports, and the workload ratings against traffic and delay data. Then tune `config.py`.
2. **Ensembles:** running the same pipeline on GEFS (Open-Meteo ensemble API) would turn "TS ≥ 1 h" into a probability, which is better for multi-day strings.
3. **HRRR for day 1:** at 3 km it resolves convection that GFS parameterises. Open-Meteo serves HRRR through the same API.
4. **Sector-level stats:** split ZJX into its areas or sectors (with a GeoJSON per sector), because workload is felt sector by sector.
5. **Automate:** use cron or a scheduled job after each GFS cycle (about 03:30, 09:30, 15:30 and 21:30Z), with images published to the briefing page.
