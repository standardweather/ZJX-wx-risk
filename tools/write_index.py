"""Write the published briefings page (index.html) and README.md.

Reads the --json outputs that sit next to the PNGs in <out_dir>, so the page
always matches the images it shows. Served by GitHub Pages from the
``briefings`` branch.

Usage: python tools/write_index.py <out_dir> [--note TEXT]
"""
import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zjxwx.config import DISCLAIMER, STYLE  # noqa: E402

TMU_COLORS = {"ts": STYLE["tmu"]["ts"], "turb_sev": STYLE["tmu"]["turb_sev"], "ice_sev": STYLE["tmu"]["ice_sev"],
              "ifr": STYLE["tmu"]["ifr"], "turb_mod": STYLE["tmu"]["turb_mod"],
              "ice_mod": STYLE["tmu"]["ice_mod"], "llws": STYLE["tmu"]["llws"]}

SHIFT_NAMES = {"F": "F shift · 0515–1315 local", "N": "N shift · 1300–2100 local"}


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def _chip(level):
    color = STYLE["cat"].get(level, "#5b6577")
    return f'<span class="chip" style="--c:{color}">{html.escape(level)}</span>'


def _day(iso):
    return datetime.fromisoformat(iso).strftime("%a %d %b")


def _tmu_chip(item):
    color = TMU_COLORS.get(item["key"], "#5b6577")
    tip = html.escape(" · ".join(x for x in (item["hours"], item["detail"]) if x))
    return f'<span class="chip" style="--c:{color}" title="{tip}">{html.escape(item["name"])}</span>'


def _tmu_section(out: Path, stamp: str):
    """OM/TMU/OS hazard & awareness maps, produced each morning by 0500L."""
    res = _load(out / "tmu" / "tmu.json")
    if not res:
        return "", []
    blocks, md = [], ["## OM/TMU/OS", ""]
    for p in res["periods"]:
        hz = [i for i in p["items"] if i["present"] and i["kind"] == "HAZARD"]
        aw = [i for i in p["items"] if i["present"] and i["kind"] == "AWARENESS"]
        none = '<span class="why">none in ZJX</span>'
        blocks.append(f"""
  <h3>{html.escape(p["name"])} · {html.escape(p["local"])}</h3>
  <p class="tmu-row"><strong>Hazard:</strong> {"".join(map(_tmu_chip, hz)) or none}</p>
  <p class="tmu-row"><strong>Awareness:</strong> {"".join(map(_tmu_chip, aw)) or none}</p>
  <a href="tmu/{p['image']}?v={stamp}"><img src="tmu/{p['image']}?v={stamp}"
     alt="OM/TMU/OS {html.escape(p['name'])} map {html.escape(p['local'])}: hazards
     {html.escape(', '.join(i['name'] for i in hz) or 'none')}; awareness
     {html.escape(', '.join(i['name'] for i in aw) or 'none')}" loading="lazy"></a>""")
        md += [f"### {p['name']} {p['local']}", "",
               "Hazard: " + (", ".join(i["name"] for i in hz) or "none") + "  ",
               "Awareness: " + (", ".join(i["name"] for i in aw) or "none"), "",
               f"![{p['name']}](tmu/{p['image']})", ""]
    issued = datetime.fromisoformat(res["issued"].replace("Z", "+00:00"))
    html_out = f"""
<section id="tmu">
  <div class="sec-head">
    <h2>OM/TMU/OS</h2>
    <p><strong>{_day(res['day'])}</strong> · issued {issued:%H%MZ}
       <span class="why">Solid = hazard · hatched = awareness</span></p>
  </div>
  {''.join(blocks)}
</section>"""
    return html_out, md


def build(out: Path, note: str = ""):
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d%H%M")
    sections, md = [], ["# ZJX pre-shift weather & workload briefings", "",
                        f"Latest images: see the GitHub Pages site. Updated {now:%Y-%m-%d %H%MZ}.", ""]
    for code in ("F", "N"):
        brief = _load(out / f"brief_{code}.json")
        string = _load(out / f"string_{code}.json")
        if not brief:
            continue
        b = brief[0]
        rows = "".join(
            f"<tr><td>{_day(r['date'])}</td><td>{_chip(r['workload'])}</td>"
            f"<td>{html.escape(r['reason'])}</td></tr>" for r in (string or []))
        table = (f'<table><thead><tr><th>Date</th><th>Workload</th><th>Driver</th></tr></thead>'
                 f"<tbody>{rows}</tbody></table>") if rows else ""
        sections.append(f"""
<section id="{code}">
  <div class="sec-head">
    <h2>{SHIFT_NAMES[code]}</h2>
    <p>Next shift: <strong>{_day(b['date'])}</strong> {_chip(b['workload'])}
       <span class="why">{html.escape(b['reason'])}</span></p>
  </div>
  <a href="brief_{code}.png?v={stamp}"><img src="brief_{code}.png?v={stamp}"
     alt="{code} shift briefing sheet for {b['date']}: workload {b['workload']}" loading="lazy"></a>
  {table}
  <a href="string_{code}.png?v={stamp}"><img src="string_{code}.png?v={stamp}"
     alt="{code} shift five-day outlook" loading="lazy"></a>
</section>""")
        md += [f"## {code} shift, {b['date']}: **{b['workload']}**", "", b["reason"], "",
               f"![{code} briefing](brief_{code}.png)", "", f"![{code} shift string](string_{code}.png)", ""]

    tmu_html, tmu_md = _tmu_section(out, stamp)
    sections.append(tmu_html)
    md += tmu_md
    nav = "".join(f'<a href="#{c}">{c} shift</a>' for c in ("F", "N") if (out / f"brief_{c}.json").exists())
    if tmu_html:
        nav += '<a href="#tmu">OM/TMU/OS</a>'
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="1800">
<title>ZJX Shift Briefings</title>
<style>
  :root {{
    --navy: {STYLE['navy']}; --ink: {STYLE['ink']}; --muted: {STYLE['muted']};
    --rule: {STYLE['rule']}; --paper: #f4f6f9; --card: #ffffff;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --ink: #e6e9ef; --muted: #a3acbb; --rule: #2c3446; --paper: #0e1320; --card: #161c2b; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--paper); color: var(--ink);
         font: 15px/1.5 Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  header {{ background: var(--navy); color: #fff; padding: 20px 16px; }}
  header .wrap, main, footer {{ max-width: 1200px; margin: 0 auto; }}
  header .kicker {{ color: #9fb3d1; font-size: 12px; font-weight: 600; letter-spacing: .04em; }}
  header h1 {{ margin: 4px 0; font-size: 26px; }}
  header .meta {{ color: #d8e1ee; font-size: 13px; }}
  nav {{ margin-top: 12px; display: flex; gap: 8px; }}
  nav a {{ color: #fff; text-decoration: none; border: 1px solid #ffffff55; border-radius: 6px; padding: 4px 12px; font-size: 13px; }}
  main {{ padding: 16px; }}
  section {{ background: var(--card); border: 1px solid var(--rule); border-radius: 10px; padding: 16px; margin-bottom: 24px; }}
  .sec-head h2 {{ margin: 0 0 4px; font-size: 20px; }}
  .sec-head p {{ margin: 0 0 12px; }}
  .why {{ color: var(--muted); font-size: 13px; margin-left: 6px; }}
  img {{ width: 100%; height: auto; display: block; border-radius: 6px; border: 1px solid var(--rule); margin: 8px 0 16px; background: #fff; }}
  .chip {{ display: inline-block; background: var(--c); color: #fff; font-weight: 700; font-size: 12px;
           border-radius: 4px; padding: 1px 8px; letter-spacing: .02em; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin: 4px 0 8px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--rule); vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12px; }}
  td:first-child {{ white-space: nowrap; }}
  footer {{ padding: 0 16px 32px; color: var(--muted); font-size: 12px; }}
  section h3 {{ margin: 18px 0 4px; font-size: 16px; }}
  .tmu-row {{ margin: 2px 0; font-size: 13px; }}
  .tmu-row .chip {{ margin-right: 4px; }}
  .note {{ background: #fff4e5; color: #6b4200; border-radius: 6px; padding: 8px 12px; margin-bottom: 16px; font-size: 13px; }}
</style>
</head>
<body>
<header><div class="wrap">
  <div class="kicker">ZJX ARTCC · PRE-SHIFT WEATHER &amp; WORKLOAD BRIEFINGS</div>
  <h1>Latest shift briefings</h1>
  <div class="meta">Updated {now:%a %d %b %Y %H%MZ} from GFS via Open-Meteo, plus NBM for the OM/TMU/OS maps · refreshes after the 00Z and 12Z model runs</div>
  <nav>{nav}</nav>
</div></header>
<main>
  {f'<div class="note">{html.escape(note)}</div>' if note else ''}
  {''.join(sections) or '<p>No briefings have been generated yet.</p>'}
</main>
<footer>{html.escape(DISCLAIMER)}</footer>
</body>
</html>
"""
    (out / "index.html").write_text(page)
    (out / ".nojekyll").write_text("")
    (out / "README.md").write_text("\n".join(md))
    print("wrote", out / "index.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    build(Path(a.out), a.note)
