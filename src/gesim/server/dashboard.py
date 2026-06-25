"""Self-contained HTML dashboard for the world-model server.

Served at ``GET /``. Polls ``/api/status`` once a second and refreshes the
generated-frame preview from ``/api/preview.jpg``. No build step, no framework.
"""

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gesim · world model</title>
<style>
  :root {
    --bg: #fafafa; --panel: #ffffff; --ink: #1a1a1a; --muted: #6b7280;
    --line: #e5e7eb; --accent: #2563eb; --ready: #2563eb; --run: #16a34a; --idle: #9ca3af;
    --pos: #16a34a; --neg: #dc2626;
    --mono: ui-monospace, "SFMono-Regular", "JetBrains Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans);
         line-height: 1.5; -webkit-font-smoothing: antialiased; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 32px 24px 64px; }
  header { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
           border-bottom: 1px solid var(--line); padding-bottom: 16px; }
  h1 { font-size: 20px; font-weight: 650; margin: 0; letter-spacing: -0.01em; }
  h1 span { color: var(--muted); font-weight: 400; }
  .chip { font-family: var(--mono); font-size: 12px; color: var(--muted);
          border: 1px solid var(--line); border-radius: 6px; padding: 2px 8px; }
  .pill { margin-left: auto; display: inline-flex; align-items: center; gap: 8px;
          font-size: 13px; font-weight: 550; padding: 5px 12px; border-radius: 999px;
          border: 1px solid var(--line); background: var(--panel); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--idle); }
  .pill.ready .dot { background: var(--ready); }
  .pill.running .dot { background: var(--run); animation: pulse 1s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

  .task { margin-top: 20px; font-size: 15px; }
  .task .lbl { color: var(--muted); font-size: 11px; text-transform: uppercase;
               letter-spacing: 0.06em; display: block; margin-bottom: 3px; }

  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 16px; }
  .stat { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
  .stat .n { font-family: var(--mono); font-size: 22px; font-weight: 600; }
  .stat .k { color: var(--muted); font-size: 12px; margin-top: 2px; }

  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
           padding: 18px 20px; margin-top: 20px; }
  .panel h2 { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
              color: var(--muted); margin: 0 0 14px; }
  .preview img { width: 100%; height: auto; display: block; border-radius: 8px;
                 border: 1px solid var(--line); background: #f3f4f6; }
  .placeholder { aspect-ratio: 4 / 1; border-radius: 8px; border: 1px dashed var(--line);
                 display: flex; align-items: center; justify-content: center;
                 color: var(--muted); font-size: 13px; background: #f3f4f6; }
  .cap { color: var(--muted); font-size: 12px; margin-top: 8px; text-align: center; }

  .cmp { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
  @media (max-width: 720px) { .cmp { grid-template-columns: 1fr; } .stats { grid-template-columns: 1fr 1fr; } }
  table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 12.5px; }
  caption { text-align: left; color: var(--muted); font-size: 11px; text-transform: uppercase;
            letter-spacing: 0.05em; padding-bottom: 6px; font-family: var(--sans); }
  th { text-align: right; color: var(--muted); font-weight: 500; font-size: 11px;
       padding: 4px 6px; border-bottom: 1px solid var(--line); }
  th:first-child { text-align: left; }
  td { text-align: right; padding: 4px 6px; border-bottom: 1px solid #f1f2f4; }
  td.dim { text-align: left; color: var(--muted); }
  td.pos { color: var(--pos); }
  td.neg { color: var(--neg); }
  .empty { color: var(--muted); font-size: 13px; font-family: var(--sans); }

  .foot { color: var(--muted); font-size: 12px; margin-top: 28px; text-align: center; }
  .foot code { font-family: var(--mono); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>gesim <span>· world model</span></h1>
    <span class="chip" id="model">—</span>
    <span class="chip" id="chunk">chunk —</span>
    <span class="pill idle" id="pill"><span class="dot"></span><span id="phase">idle</span></span>
  </header>

  <div class="task"><span class="lbl">task</span><span id="task">—</span></div>

  <div class="stats">
    <div class="stat"><div class="n" id="steps">0</div><div class="k">steps</div></div>
    <div class="stat"><div class="n" id="frames">0</div><div class="k">frames generated</div></div>
    <div class="stat"><div class="n" id="uptime">—</div><div class="k">uptime</div></div>
  </div>

  <section class="panel preview">
    <h2>Latest generated frame</h2>
    <img id="preview" alt="generated frame" style="display:none">
    <div class="placeholder" id="noprev">no frame generated yet</div>
    <div class="cap">head&nbsp;·&nbsp;left wrist&nbsp;·&nbsp;right wrist</div>
  </section>

  <section class="panel">
    <h2>Action vs predicted state &nbsp;<span style="color:var(--muted);text-transform:none;letter-spacing:0;font-weight:400">(commanded action vs world-model prediction, last frame)</span></h2>
    <div class="cmp" id="cmp"><div class="empty">waiting for a step…</div></div>
  </section>

  <div class="foot">polling <code>/api/status</code> every second · <code>/healthz</code> for liveness</div>
</div>

<script>
const $ = (id) => document.getElementById(id);
let lastStep = -1;

function fmtUptime(s) {
  s = Math.floor(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return (h ? h + "h " : "") + (h || m ? m + "m " : "") + sec + "s";
}

const LABELS = ["L1","L2","L3","L4","L5","L6","L7","L grip","R1","R2","R3","R4","R5","R6","R7","R grip"];

function tableFor(title, action, state, lo, hi) {
  let rows = "";
  for (let i = lo; i < hi; i++) {
    const a = action[i], p = state[i], d = p - a;
    const cls = Math.abs(d) < 1e-4 ? "" : (d > 0 ? "pos" : "neg");
    rows += "<tr><td class='dim'>" + LABELS[i] + "</td><td>" + a.toFixed(3) +
            "</td><td>" + p.toFixed(3) + "</td><td class='" + cls + "'>" +
            (d >= 0 ? "+" : "") + d.toFixed(3) + "</td></tr>";
  }
  return "<table><caption>" + title + "</caption><thead><tr><th>dim</th><th>action</th>" +
         "<th>predicted</th><th>Δ</th></tr></thead><tbody>" + rows + "</tbody></table>";
}

function renderCmp(action, state) {
  const el = $("cmp");
  if (!action || !state) { el.innerHTML = '<div class="empty">waiting for a step…</div>'; return; }
  el.innerHTML = tableFor("left arm", action, state, 0, 8) +
                 tableFor("right arm", action, state, 8, 16);
}

async function tick() {
  let s;
  try { s = await (await fetch("/api/status", {cache: "no-store"})).json(); }
  catch (e) { $("phase").textContent = "unreachable"; return; }

  $("model").textContent = s.model;
  $("chunk").textContent = "chunk " + s.chunk_size;
  $("phase").textContent = s.phase;
  $("pill").className = "pill " + s.phase;
  $("task").textContent = s.task || "—";
  $("steps").textContent = s.step_count;
  $("frames").textContent = s.frames_generated;
  $("uptime").textContent = fmtUptime(s.uptime_s);
  renderCmp(s.action, s.state);

  if (s.has_preview && s.step_count !== lastStep) {
    lastStep = s.step_count;
    const img = $("preview");
    img.onload = () => { img.style.display = "block"; $("noprev").style.display = "none"; };
    img.src = "/api/preview.jpg?t=" + s.step_count;
  }
}

tick();
setInterval(tick, 1000);
</script>
</body>
</html>"""
