"""Export a validated scenario to a runtime scene description and a viewer.

``scene_unity.json`` is the flat contract a Unity importer consumes: one record
per instance with a transform, an asset key and the hazard metadata the runtime
needs to score a trainee's response. Keeping it flat means the game engine never
has to understand the scenario graph.

The viewer embeds its data inline and uses no external libraries, so the output
opens straight from disk with no server and no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .schema import Scenario
from .site import SiteModel
from .validate import ValidationReport

ASSET_PREFAB = {
    "guardrail_system": "Prefabs/Safety/Guardrail",
    "portable_ladder": "Prefabs/Access/ExtensionLadder",
    "scaffold_platform": "Prefabs/Access/ScaffoldBay",
    "spoil_pile": "Prefabs/Earthworks/SpoilPile",
    "mobile_crane": "Prefabs/Plant/MobileCrane",
    "haul_truck": "Prefabs/Plant/HaulTruck",
    "rebar_cage": "Prefabs/Concrete/RebarCage",
    "floor_opening": "Prefabs/Structure/FloorOpening",
    "deck_edge": "Prefabs/Structure/DeckEdge",
    "overhead_power_line": "Prefabs/Utilities/OverheadLine",
    "suspended_load": "Prefabs/Rigging/SuspendedLoad",
}
DEFAULT_PREFAB = "Prefabs/Generic/Placeholder"


def to_unity_scene(scenario: Scenario, site: SiteModel, report: ValidationReport) -> Dict:
    hazard_by_entity = {h.target_entity: h for h in scenario.hazards}
    instances = []
    for e in scenario.entities:
        haz = hazard_by_entity.get(e.id)
        instances.append(
            {
                "instanceId": e.id,
                "prefab": ASSET_PREFAB.get(e.asset_type, DEFAULT_PREFAB),
                "assetType": e.asset_type,
                "kind": e.kind,
                "transform": {
                    "position": [round(float(e.x or 0.0), 2), round(e.level_ft * 0.3048, 2),
                                 round(float(e.y or 0.0), 2)],
                    "rotationY": 0.0,
                },
                "zone": e.zone,
                "isDistractor": e.is_distractor,
                "params": {k: v for k, v in e.params.items() if k != "controls"},
                "controls": e.params.get("controls", []),
                "scoring": None
                if haz is None
                else {
                    "hazardId": haz.id,
                    "hazardClass": haz.hazard_class,
                    "clause": haz.clause,
                    "severity": haz.severity,
                    "requiredControls": haz.required_controls,
                    "cueSalience": haz.cue_salience,
                },
            }
        )

    return {
        "formatVersion": "safesitegen/scene-1.0",
        "scenarioId": scenario.id,
        "site": site.to_dict(),
        "trade": scenario.trade,
        "activity": scenario.activity,
        "difficulty": round(report.difficulty, 3),
        "validated": report.passed,
        "instances": instances,
        "answerKey": [
            {"hazardId": h.id, "entityId": h.target_entity, "hazardClass": h.hazard_class,
             "clause": h.clause}
            for h in scenario.hazards
        ],
    }


# --------------------------------------------------------------------------- #
# Viewer
# --------------------------------------------------------------------------- #

_VIEWER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SafeSiteGen scenario __SCENARIO_ID__</title>
<style>
  :root { --ink:#16202b; --muted:#6b7785; --line:#dde3ea; --bg:#f6f8fa; --hazard:#c8322f;
          --distract:#2f6fc8; --ok:#1e7a4c; --route:#8a6d3b; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
         color:var(--ink); background:var(--bg); }
  header { padding:16px 22px; background:#fff; border-bottom:1px solid var(--line); }
  h1 { margin:0 0 3px; font-size:16px; letter-spacing:.2px; }
  .sub { color:var(--muted); font-size:12.5px; }
  .wrap { display:flex; gap:18px; padding:18px 22px; align-items:flex-start; flex-wrap:wrap; }
  .panel { background:#fff; border:1px solid var(--line); border-radius:8px; padding:14px 16px; }
  #stage { flex:1 1 620px; min-width:420px; }
  #side { flex:0 1 380px; min-width:320px; max-height:76vh; overflow:auto; }
  canvas { width:100%; height:auto; display:block; border-radius:5px; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.9px; color:var(--muted);
       margin:0 0 9px; font-weight:600; }
  .chk { padding:7px 0; border-bottom:1px dotted var(--line); font-size:12.5px; }
  .chk:last-child { border-bottom:0; }
  .tag { display:inline-block; font-size:10.5px; font-weight:700; padding:1px 6px;
         border-radius:3px; margin-right:7px; vertical-align:1px; }
  .pass { background:#e4f3ea; color:var(--ok); }
  .fail { background:#fbe6e5; color:var(--hazard); }
  .cid { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11.5px; color:var(--ink); }
  .det { color:var(--muted); }
  .legend { display:flex; gap:16px; flex-wrap:wrap; margin-top:10px; font-size:12px; color:var(--muted); }
  .swatch { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  td { padding:4px 0; vertical-align:top; }
  td.k { color:var(--muted); width:44%; }
  .clause { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11.5px; }
</style>
</head>
<body>
<header>
  <h1>__SCENARIO_ID__ &middot; __SITE_NAME__</h1>
  <div class="sub">__TRADE__ &middot; __ACTIVITY__ &middot; difficulty __DIFFICULTY__ &middot; __VERDICT__</div>
</header>
<div class="wrap">
  <div class="panel" id="stage">
    <h2>Site plan</h2>
    <canvas id="c" width="1000" height="620"></canvas>
    <div class="legend">
      <span><i class="swatch" style="background:var(--hazard)"></i>teaching point</span>
      <span><i class="swatch" style="background:var(--distract)"></i>distractor</span>
      <span><i class="swatch" style="background:#94a3b2"></i>context / plant</span>
      <span><i class="swatch" style="background:var(--route)"></i>trainee route</span>
      <span><i class="swatch" style="background:#c9ced6"></i>fixed geometry</span>
    </div>
  </div>
  <div class="panel" id="side">
    <h2>Teaching points</h2>
    <div id="haz"></div>
    <h2 style="margin-top:18px">Validation report</h2>
    <div id="checks"></div>
  </div>
</div>
<script>
const DATA = __DATA__;

const ZONE_FILL = {
  slab_edge:"#fdf0e6", deck:"#f2f5f8", access:"#eef4ee", laydown:"#f7f4ec",
  excavation:"#efeae2", work_zone:"#eef1f7", haul_road:"#eceff2"
};

function draw() {
  const cv = document.getElementById("c"), g = cv.getContext("2d");
  const site = DATA.scene.site;
  const pad = 40;
  const s = Math.min((cv.width - 2*pad) / site.width, (cv.height - 2*pad) / site.depth);
  const X = x => pad + x * s, Y = y => pad + y * s;

  g.clearRect(0,0,cv.width,cv.height);
  g.fillStyle = "#fff"; g.fillRect(0,0,cv.width,cv.height);

  for (const z of site.zones) {
    g.fillStyle = ZONE_FILL[z.type] || "#f4f6f8";
    g.fillRect(X(z.x), Y(z.y), z.w*s, z.d*s);
    g.strokeStyle = "#e2e7ec"; g.lineWidth = 1;
    g.strokeRect(X(z.x), Y(z.y), z.w*s, z.d*s);
    g.fillStyle = "#98a3ae"; g.font = "10px Helvetica";
    g.fillText(z.type, X(z.x)+4, Y(z.y)+12);
  }

  g.strokeStyle = "#eef1f4"; g.lineWidth = 0.5;
  for (let x=0; x<=site.width; x++){ g.beginPath(); g.moveTo(X(x),Y(0)); g.lineTo(X(x),Y(site.depth)); g.stroke(); }
  for (let y=0; y<=site.depth; y++){ g.beginPath(); g.moveTo(X(0),Y(y)); g.lineTo(X(site.width),Y(y)); g.stroke(); }

  for (const o of site.obstacles) {
    g.fillStyle = "#c9ced6";
    g.fillRect(X(o.x), Y(o.y), o.w*s, o.d*s);
  }

  g.strokeStyle = "#8a6d3b"; g.lineWidth = 1.6; g.setLineDash([5,4]);
  g.beginPath();
  site.patrol.forEach((p,i) => i ? g.lineTo(X(p[0]+0.5),Y(p[1]+0.5)) : g.moveTo(X(p[0]+0.5),Y(p[1]+0.5)));
  g.stroke(); g.setLineDash([]);
  site.patrol.forEach(p => {
    g.fillStyle = "#8a6d3b"; g.beginPath();
    g.arc(X(p[0]+0.5), Y(p[1]+0.5), 3, 0, 6.284); g.fill();
  });

  const keyed = {};
  DATA.scene.answerKey.forEach((a,i) => keyed[a.entityId] = i+1);

  for (const inst of DATA.scene.instances) {
    const px = X(inst.transform.position[0] + 0.5), py = Y(inst.transform.position[2] + 0.5);
    const isHaz = inst.scoring !== null;
    g.fillStyle = isHaz ? "#c8322f" : (inst.isDistractor ? "#2f6fc8" : "#94a3b2");
    g.beginPath(); g.arc(px, py, isHaz ? 8 : 5.5, 0, 6.284); g.fill();
    if (isHaz) {
      g.strokeStyle = "rgba(200,50,47,.35)"; g.lineWidth = 1.4;
      g.beginPath(); g.arc(px, py, 15, 0, 6.284); g.stroke();
      g.fillStyle = "#fff"; g.font = "bold 10px Helvetica"; g.textAlign = "center";
      g.fillText(String(keyed[inst.instanceId] || ""), px, py+3.5);
      g.textAlign = "start";
    }
    g.fillStyle = "#5c6773"; g.font = "9.5px Helvetica";
    g.fillText(inst.assetType, px + 11, py + 3);
  }
}

function render() {
  const hz = document.getElementById("haz");
  hz.innerHTML = DATA.scene.answerKey.map((a,i) => {
    const inst = DATA.scene.instances.find(x => x.instanceId === a.entityId) || {};
    const req = (inst.scoring && inst.scoring.requiredControls || []).join(", ") || "n/a";
    return `<div class="chk"><span class="tag fail">${i+1}</span>
      <b>${a.hazardClass.replace(/_/g," ")}</b><br/>
      <table>
        <tr><td class="k">clause</td><td class="clause">${a.clause}</td></tr>
        <tr><td class="k">required control</td><td>${req}</td></tr>
        <tr><td class="k">present control</td><td>${(inst.controls||[]).join(", ") || "none"}</td></tr>
        <tr><td class="k">location</td><td>${inst.transform ?
          inst.transform.position[0]+", "+inst.transform.position[2] : "?"}</td></tr>
      </table></div>`;
  }).join("");

  document.getElementById("checks").innerHTML = DATA.report.checks.map(c =>
    `<div class="chk"><span class="tag ${c.status}">${c.status}</span>
     <span class="cid">${c.check_id}</span><br/>
     <span class="det">${c.detail}</span></div>`).join("");
}

draw(); render();
</script>
</body>
</html>
"""


def write_viewer(path: Path, scenario: Scenario, site: SiteModel, report: ValidationReport) -> Path:
    scene = to_unity_scene(scenario, site, report)
    payload = json.dumps({"scene": scene, "report": report.to_dict()})
    html = (
        _VIEWER_TEMPLATE
        .replace("__DATA__", payload)
        .replace("__SCENARIO_ID__", scenario.id)
        .replace("__SITE_NAME__", site.name)
        .replace("__TRADE__", scenario.trade.replace("_", " "))
        .replace("__ACTIVITY__", scenario.activity)
        .replace("__DIFFICULTY__", f"{report.difficulty:.2f}")
        .replace("__VERDICT__", "validated" if report.passed else "rejected by the gate")
    )
    path.write_text(html, encoding="utf-8")
    return path


def export_all(
    scenario: Scenario,
    report: ValidationReport,
    out_dir: str | Path,
    site: Optional[SiteModel] = None,
) -> Dict[str, Path]:
    site = site or SiteModel.load(scenario.site_template)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "scenario": out / "scenario.json",
        "report": out / "report.json",
        "scene": out / "scene_unity.json",
        "viewer": out / "viewer.html",
    }
    paths["scenario"].write_text(scenario.to_json(), encoding="utf-8")
    paths["report"].write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    paths["scene"].write_text(json.dumps(to_unity_scene(scenario, site, report), indent=2), encoding="utf-8")
    write_viewer(paths["viewer"], scenario, site, report)
    return paths
