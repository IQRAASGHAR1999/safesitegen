"""First-person 3D training environment.

``export.write_viewer`` produces a plan view for inspecting a scenario. This
module produces the thing a trainee would actually use: a walkable site rendered
in real time, in which the trainee looks around, clicks on what they believe is
a hazard, and is scored against the answer key the validation gate already
verified.

It is a single self-contained HTML file with the scene embedded inline, no
libraries, no build step, no network. Rendering is a hand-written painter's
algorithm over the same ``scene_unity.json`` contract the Unity importer reads,
which is the point: the runtime contract is engine agnostic, and this is one
renderer of it rather than the renderer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .export import to_unity_scene
from .schema import Scenario
from .site import SiteModel
from .validate import ValidationReport

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SafeSiteGen training environment &middot; __SCENARIO_ID__</title>
<style>
  :root { --ink:#101820; --muted:#7c8794; --line:#2b3642; --hazard:#e0453f;
          --ok:#37b26d; --warn:#e8a33d; --panel:#141c24; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; overflow:hidden; background:#0b1117;
              font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
              color:#e7edf3; }
  #view { position:absolute; inset:0; cursor:crosshair; display:block; }
  .hud { position:absolute; background:rgba(20,28,36,.9); border:1px solid var(--line);
         border-radius:8px; padding:12px 14px; backdrop-filter:blur(6px); }
  #top { top:14px; left:14px; max-width:430px; }
  #top h1 { margin:0 0 3px; font-size:14px; letter-spacing:.2px; }
  #top .meta { color:var(--muted); font-size:12px; }
  #score { top:14px; right:14px; text-align:right; min-width:180px; }
  #score .n { font-size:26px; font-weight:700; line-height:1.1; }
  #score .l { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.9px; }
  #help { bottom:14px; left:14px; color:var(--muted); font-size:12px; }
  #help kbd { background:#1e2831; border:1px solid var(--line); border-radius:3px;
              padding:1px 5px; font-family:ui-monospace,Menlo,monospace; font-size:11px;
              color:#cdd8e2; }
  #log { bottom:14px; right:14px; width:340px; max-height:44vh; overflow:auto; }
  #log h2 { margin:0 0 8px; font-size:11px; text-transform:uppercase; letter-spacing:.9px;
            color:var(--muted); font-weight:600; }
  .entry { padding:7px 0; border-bottom:1px dotted var(--line); font-size:12.5px; }
  .entry:last-child { border-bottom:0; }
  .entry .t { font-weight:600; }
  .entry .c { font-family:ui-monospace,Menlo,monospace; font-size:11px; color:var(--muted); }
  .hit .t { color:var(--ok); }
  .miss .t { color:var(--hazard); }
  .fp .t { color:var(--warn); }
  #done { position:absolute; inset:0; display:none; align-items:center; justify-content:center;
          background:rgba(7,11,15,.9); z-index:20; }
  #done .card { background:var(--panel); border:1px solid var(--line); border-radius:12px;
                padding:26px 30px; max-width:540px; }
  #done h2 { margin:0 0 4px; font-size:19px; }
  #done .sub { color:var(--muted); font-size:12.5px; margin-bottom:16px; }
  #done table { width:100%; border-collapse:collapse; font-size:13px; }
  #done td { padding:5px 0; border-bottom:1px dotted var(--line); }
  #done td.r { text-align:right; color:var(--muted); font-family:ui-monospace,Menlo,monospace; }
  button { margin-top:18px; background:#22303d; color:#e7edf3; border:1px solid var(--line);
           border-radius:6px; padding:8px 16px; font-size:13px; cursor:pointer; }
  button:hover { background:#2b3b4a; }
  #crosshair { position:absolute; left:50%; top:50%; width:16px; height:16px; margin:-8px 0 0 -8px;
               pointer-events:none; }
  #crosshair:before, #crosshair:after { content:""; position:absolute; background:rgba(255,255,255,.55); }
  #crosshair:before { left:7px; top:0; width:2px; height:16px; }
  #crosshair:after { top:7px; left:0; height:2px; width:16px; }
</style>
</head>
<body>
<canvas id="view"></canvas>
<div id="crosshair"></div>

<div class="hud" id="top">
  <h1>__SCENARIO_ID__</h1>
  <div class="meta">__SITE_NAME__ &middot; __TRADE__, __ACTIVITY__<br/>
  Walk the site and click every hazard you can find.</div>
</div>

<div class="hud" id="score">
  <div class="n"><span id="found">0</span> / <span id="total">0</span></div>
  <div class="l">hazards found</div>
</div>

<div class="hud" id="help">
  <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move &nbsp;
  <kbd>drag</kbd> look &nbsp; <kbd>click</kbd> flag a hazard &nbsp;
  <kbd>Enter</kbd> finish
</div>

<div class="hud" id="log"><h2>Your report</h2><div id="entries"></div></div>

<div id="done"><div class="card">
  <h2 id="verdict"></h2>
  <div class="sub" id="verdictSub"></div>
  <table id="breakdown"></table>
  <button onclick="location.reload()">Walk it again</button>
</div></div>

<script>
const DATA = __DATA__;
const scene = DATA.scene, site = scene.site;

/* ---------------------------------------------------------------- state */
const cam = { x: site.entry[0] + 0.5, y: 1.68, z: site.entry[1] + 0.5, yaw: 0, pitch: 0 };
const keys = {};
const flagged = new Set();
const answer = {};
scene.answerKey.forEach(a => answer[a.entityId] = a);
let finished = false;

document.getElementById("total").textContent = scene.answerKey.length;

/* ------------------------------------------------------------- geometry */
const PALETTE = {
  worker:     { c: "#e8b84b", h: 1.75, w: 0.55 },
  equipment:  { c: "#7fa6c9", h: 2.20, w: 1.30 },
  structure:  { c: "#a2adb8", h: 1.10, w: 1.10 },
  control:    { c: "#6fbf8f", h: 1.05, w: 1.60 },
  environment:{ c: "#b79ccf", h: 4.50, w: 0.60 }
};
const ZONE_COLOUR = {
  slab_edge:"#3a3228", deck:"#2a323a", access:"#28332c", laydown:"#33302a",
  excavation:"#2e2823", work_zone:"#262e38", scaffold_bay:"#332b33", haul_road:"#242a30"
};

function project(wx, wy, wz, W, H) {
  const dx = wx - cam.x, dy = wy - cam.y, dz = wz - cam.z;
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  let rx = dx * cy - dz * sy;
  let rz = dx * sy + dz * cy;
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  let ry = dy * cp - rz * sp;
  rz = dy * sp + rz * cp;
  if (rz < 0.12) return null;
  const f = (H * 0.9) / rz;
  return { x: W / 2 + rx * f, y: H / 2 - ry * f, d: rz, s: f };
}

function shade(hex, k) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.min(255, Math.round(((n >> 16) & 255) * k));
  const g = Math.min(255, Math.round(((n >> 8) & 255) * k));
  const b = Math.min(255, Math.round((n & 255) * k));
  return `rgb(${r},${g},${b})`;
}

/* ----------------------------------------------------------------- draw */
const cv = document.getElementById("view"), g = cv.getContext("2d");
function resize() { cv.width = innerWidth; cv.height = innerHeight; }
addEventListener("resize", resize); resize();

let pickable = [];

function draw() {
  const W = cv.width, H = cv.height;

  const sky = g.createLinearGradient(0, 0, 0, H);
  sky.addColorStop(0, "#16212c"); sky.addColorStop(0.55, "#2a3a48"); sky.addColorStop(1, "#1a232c");
  g.fillStyle = sky; g.fillRect(0, 0, W, H);

  const quads = [];

  for (const z of site.zones) {
    const pts = [[z.x, z.y], [z.x + z.w, z.y], [z.x + z.w, z.y + z.d], [z.x, z.y + z.d]]
      .map(p => project(p[0], 0, p[1], W, H));
    if (pts.some(p => !p)) continue;
    quads.push({ d: Math.max(...pts.map(p => p.d)), kind: "zone",
                 pts, fill: ZONE_COLOUR[z.type] || "#2a323a", label: z.type });
  }

  for (const o of site.obstacles) {
    quads.push({ d: dist(o.x + o.w / 2, o.y + o.d / 2), kind: "box",
                 x: o.x + o.w / 2, z: o.y + o.d / 2, w: Math.max(o.w, o.d), h: 1.9,
                 colour: "#4d5761", label: o.asset_type, id: null });
  }

  for (const inst of scene.instances) {
    const p = PALETTE[inst.kind] || PALETTE.structure;
    const x = inst.transform.position[0] + 0.5, z = inst.transform.position[2] + 0.5;
    quads.push({ d: dist(x, z), kind: "box", x, z, w: p.w, h: p.h,
                 colour: p.c, label: inst.assetType, id: inst.instanceId,
                 hazard: inst.scoring !== null });
  }

  quads.sort((a, b) => b.d - a.d);
  pickable = [];

  for (const q of quads) {
    if (q.kind === "zone") {
      g.beginPath();
      q.pts.forEach((p, i) => i ? g.lineTo(p.x, p.y) : g.moveTo(p.x, p.y));
      g.closePath();
      g.fillStyle = q.fill; g.fill();
      g.strokeStyle = "rgba(255,255,255,.07)"; g.lineWidth = 1; g.stroke();
    } else {
      drawBox(q, W, H);
    }
  }
}

function dist(x, z) { return Math.hypot(x - cam.x, z - cam.z); }

function drawBox(q, W, H) {
  const half = q.w / 2;
  const base = project(q.x, 0, q.z, W, H);
  const top = project(q.x, q.h, q.z, W, H);
  if (!base || !top) return;

  const wpx = half * base.s * 2;
  const hpx = base.y - top.y;
  const x0 = base.x - wpx / 2, y0 = top.y;

  const fog = Math.max(0.35, Math.min(1, 1 - q.d / 52));
  g.fillStyle = shade(q.colour, 0.55 * fog + 0.25);
  g.fillRect(x0, y0, wpx, hpx);
  g.fillStyle = shade(q.colour, 0.85 * fog + 0.25);
  g.fillRect(x0, y0, wpx, Math.max(2, hpx * 0.18));

  if (q.id) {
    const done = flagged.has(q.id);
    if (done) {
      g.strokeStyle = answer[q.id] ? "#37b26d" : "#e8a33d";
      g.lineWidth = 2.5; g.strokeRect(x0 - 3, y0 - 3, wpx + 6, hpx + 6);
    }
    pickable.push({ id: q.id, x0, y0, x1: x0 + wpx, y1: y0 + hpx, d: q.d });
  }

  if (q.d < 26) {
    g.fillStyle = `rgba(214,226,238,${Math.min(0.75, fog)})`;
    g.font = "11px Helvetica";
    g.textAlign = "center";
    g.fillText(q.label.replace(/_/g, " "), base.x, y0 - 7);
    g.textAlign = "start";
  }
}

/* --------------------------------------------------------------- input */
let dragging = false, lastX = 0, lastY = 0, moved = 0;

cv.addEventListener("mousedown", e => { dragging = true; moved = 0; lastX = e.clientX; lastY = e.clientY; });
addEventListener("mouseup", e => {
  if (dragging && moved < 5) pick(e.clientX, e.clientY);
  dragging = false;
});
addEventListener("mousemove", e => {
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  moved += Math.abs(dx) + Math.abs(dy);
  cam.yaw -= dx * 0.004;
  cam.pitch = Math.max(-0.9, Math.min(0.9, cam.pitch - dy * 0.004));
  lastX = e.clientX; lastY = e.clientY;
});
addEventListener("keydown", e => {
  keys[e.key.toLowerCase()] = true;
  if (e.key === "Enter") finish();
});
addEventListener("keyup", e => keys[e.key.toLowerCase()] = false);

function step() {
  if (!finished) {
    const sp = keys["shift"] ? 0.18 : 0.09;
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    let fx = 0, fz = 0;
    if (keys["w"] || keys["arrowup"]) { fx += cy; fz -= sy; }
    if (keys["s"] || keys["arrowdown"]) { fx -= cy; fz += sy; }
    if (keys["a"] || keys["arrowleft"]) { fx -= sy; fz -= cy; }
    if (keys["d"] || keys["arrowright"]) { fx += sy; fz += cy; }
    const m = Math.hypot(fx, fz);
    if (m > 0) {
      cam.x = Math.max(-4, Math.min(site.width + 4, cam.x + (fx / m) * sp));
      cam.z = Math.max(-4, Math.min(site.depth + 4, cam.z + (fz / m) * sp));
    }
  }
  draw();
  requestAnimationFrame(step);
}

function pick(px, py) {
  if (finished) return;
  const hits = pickable
    .filter(p => px >= p.x0 && px <= p.x1 && py >= p.y0 && py <= p.y1)
    .sort((a, b) => a.d - b.d);
  if (!hits.length) return;
  const id = hits[0].id;
  if (flagged.has(id)) return;
  flagged.add(id);
  logEntry(id);
}

function logEntry(id) {
  const inst = scene.instances.find(i => i.instanceId === id);
  const a = answer[id];
  const box = document.createElement("div");
  box.className = "entry " + (a ? "hit" : "fp");
  box.innerHTML = a
    ? `<div class="t">Hazard: ${a.hazardClass.replace(/_/g, " ")}</div>
       <div class="c">${a.clause}</div>`
    : `<div class="t">Not a hazard: ${inst.assetType.replace(/_/g, " ")}</div>
       <div class="c">this one is compliant</div>`;
  document.getElementById("entries").prepend(box);
  document.getElementById("found").textContent =
    [...flagged].filter(f => answer[f]).length;
}

function finish() {
  if (finished) return;
  finished = true;
  const hits = [...flagged].filter(f => answer[f]);
  const fps = [...flagged].filter(f => !answer[f]);
  const missed = scene.answerKey.filter(a => !flagged.has(a.entityId));

  document.getElementById("verdict").textContent =
    missed.length === 0 ? "All hazards identified" : `${missed.length} hazard(s) missed`;
  document.getElementById("verdictSub").textContent =
    "Every entry below is traceable to the clause the validation gate verified before this scene was built.";

  const rows = [];
  hits.forEach(id => rows.push([`Found: ${answer[id].hazardClass.replace(/_/g, " ")}`, answer[id].clause]));
  missed.forEach(a => rows.push([`Missed: ${a.hazardClass.replace(/_/g, " ")}`, a.clause]));
  fps.forEach(id => {
    const inst = scene.instances.find(i => i.instanceId === id);
    rows.push([`False alarm: ${inst.assetType.replace(/_/g, " ")}`, "compliant"]);
  });
  document.getElementById("breakdown").innerHTML =
    rows.map(r => `<tr><td>${r[0]}</td><td class="r">${r[1]}</td></tr>`).join("");
  document.getElementById("done").style.display = "flex";

  const payload = {
    scenarioId: scene.scenarioId,
    detected: hits.map(id => answer[id].hazardClass),
    missed: missed.map(a => a.hazardClass),
    falsePositives: fps.length
  };
  console.log("TRAINEE_RESULT " + JSON.stringify(payload));
}

step();
</script>
</body>
</html>
"""


def write_environment(
    path: Path,
    scenario: Scenario,
    site: SiteModel,
    report: ValidationReport,
) -> Path:
    """Write the walkable training environment for one validated scenario."""
    scene = to_unity_scene(scenario, site, report)
    payload = json.dumps({"scene": scene, "report": report.to_dict()})
    html = (
        _TEMPLATE
        .replace("__DATA__", payload)
        .replace("__SCENARIO_ID__", scenario.id)
        .replace("__SITE_NAME__", site.name)
        .replace("__TRADE__", scenario.trade.replace("_", " "))
        .replace("__ACTIVITY__", scenario.activity)
    )
    path.write_text(html, encoding="utf-8")
    return path
