"""Write README.md for the published briefings branch from the --json outputs.

Usage: python tools/write_index.py <out_dir>
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
lines = ["# ZJX pre-shift weather & workload briefings", "",
         f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H%MZ} from GFS via Open-Meteo. "
         "Unofficial planning aid; see the disclaimer on each image.", ""]
for code in ("F", "N"):
    brief = out / f"brief_{code}.json"
    if not brief.exists():
        continue
    b = json.loads(brief.read_text())[0]
    lines += [f"## {code} shift, {b['date']}: **{b['workload']}**", "", b["reason"], "",
              f"![{code} briefing](brief_{code}.png)", ""]
    string = out / f"string_{code}.json"
    if string.exists():
        rows = json.loads(string.read_text())
        lines += ["| Date | Workload | Why |", "|---|---|---|"]
        lines += [f"| {r['date']} | {r['workload']} | {r['reason']} |" for r in rows]
        lines += ["", f"![{code} shift string](string_{code}.png)", ""]
(out / "README.md").write_text("\n".join(lines))
print("wrote", out / "README.md")
