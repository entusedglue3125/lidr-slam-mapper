"""
Interactive 2D LiDAR SLAM & Room Mapping Application.
Features:
- Live 2D Occupancy Grid Mapping (Log-odds + Bresenham ray-tracing)
- Real-time 2D ICP Scan Matching (Odometry & Pose Tracking)
- Web UI on http://localhost:8080 with Pan, Zoom, Trajectory, and Measurement Tool
- Export Maps as High-Res PNG or standard ROS format (PGM + YAML)
"""

import io
import os
import sys
import json
import time
import math
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
import numpy as np

from ydlidar import YDLidar
from mapper import LidarSLAM

# Output directory for saved maps
MAPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")
os.makedirs(MAPS_DIR, exist_ok=True)

# Shared State
state_lock = threading.Lock()
slam = LidarSLAM(size_cells=600, resolution_m=0.03)  # 600x600 cells @ 3cm = 18m x 18m area
latest_live_pts = []
latest_fps = 0.0
latest_inlier = 1.0
running = True
paused = False


def lidar_slam_thread():
    global latest_live_pts, latest_fps, latest_inlier, running, paused
    scan_count = 0
    t0 = time.time()

    while running:
        try:
            with YDLidar() as lidar:
                print("[SLAM Worker] Connected to YDLIDAR on /dev/ttyUSB0.")
                for scan in lidar.iter_scans():
                    if not running:
                        break
                    if paused:
                        time.sleep(0.05)
                        continue

                    scan_count += 1
                    now = time.time()
                    if now - t0 >= 1.0:
                        latest_fps = round(scan_count / (now - t0), 1)
                        scan_count = 0
                        t0 = now

                    with state_lock:
                        world_pts, inlier = slam.process_scan(scan)
                        latest_live_pts = world_pts.tolist()
                        latest_inlier = round(inlier, 2)
        except Exception as e:
            if running:
                print(f"[SLAM Worker] Serial connection error ({e}). Retrying in 1s...")
                time.sleep(1.0)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LiDAR 2D SLAM & Room Mapper</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: #090d16; color: #f1f5f9; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  
  /* Header */
  header {
    background: #111827;
    padding: 10px 20px;
    border-bottom: 1px solid #1f2937;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
  }
  .title-group { display: flex; align-items: center; gap: 10px; }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #10b981; box-shadow: 0 0 10px #10b981; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.9); } }
  h1 { font-size: 1.15rem; font-weight: 700; color: #38bdf8; letter-spacing: -0.01em; }
  
  .stats-container { display: flex; gap: 12px; font-size: 0.82rem; }
  .stat-chip { background: #1f2937; border: 1px solid #374151; padding: 4px 10px; border-radius: 6px; display: flex; flex-direction: column; }
  .stat-chip span:first-child { color: #94a3b8; font-size: 0.7rem; text-transform: uppercase; font-weight: 600; }
  .stat-chip span:last-child { color: #f8fafc; font-weight: 700; font-family: monospace; font-size: 0.9rem; }

  /* Controls Bar */
  .toolbar {
    background: #0f172a;
    padding: 8px 20px;
    border-bottom: 1px solid #1e293b;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    font-size: 0.85rem;
  }
  .btn {
    background: #1e293b;
    border: 1px solid #334155;
    color: #e2e8f0;
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s ease;
  }
  .btn:hover { background: #334155; border-color: #475569; }
  .btn:active { transform: scale(0.98); }
  .btn.active { background: #0284c7; border-color: #38bdf8; color: white; }
  .btn.danger:hover { background: #991b1b; border-color: #dc2626; color: white; }
  .btn.success { background: #065f46; border-color: #059669; }
  .btn.success:hover { background: #047857; }

  .divider { width: 1px; height: 22px; background: #334155; margin: 0 4px; }
  
  /* Canvas Viewport */
  main {
    flex: 1;
    position: relative;
    overflow: hidden;
    background: #030712;
    cursor: grab;
  }
  main.measuring { cursor: crosshair; }
  main:active { cursor: grabbing; }
  canvas { display: block; width: 100%; height: 100%; }

  /* Floating Overlay HUD */
  .hud {
    position: absolute;
    bottom: 16px;
    right: 16px;
    background: rgba(15, 23, 42, 0.88);
    backdrop-filter: blur(8px);
    border: 1px solid #334155;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 0.8rem;
    line-height: 1.5;
    color: #94a3b8;
    pointer-events: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  }
  .hud b { color: #f1f5f9; }
  .hud-legend { display: flex; gap: 12px; margin-top: 4px; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-color { width: 10px; height: 10px; border-radius: 2px; }

  /* Notification Toast */
  .toast {
    position: absolute;
    top: 16px;
    left: 50%;
    transform: translateX(-50%);
    background: #059669;
    color: white;
    padding: 8px 18px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
    z-index: 100;
  }
  .toast.show { opacity: 1; }
</style>
</head>
<body>

<header>
  <div class="title-group">
    <div class="status-dot"></div>
    <h1>LiDAR 2D SLAM & Floor Plan Mapper</h1>
  </div>
  <div class="stats-container">
    <div class="stat-chip"><span>Rate</span><span id="st-fps">-- Hz</span></div>
    <div class="stat-chip"><span>Position (X, Y)</span><span id="st-pose">0.00m, 0.00m</span></div>
    <div class="stat-chip"><span>Heading</span><span id="st-heading">0.0°</span></div>
    <div class="stat-chip"><span>Distance</span><span id="st-dist">0.00 m</span></div>
    <div class="stat-chip"><span>Match Quality</span><span id="st-inlier">--%</span></div>
  </div>
</header>

<div class="toolbar">
  <!-- Mode buttons -->
  <button id="btnSlam" class="btn active" onclick="setMode('slam')">🚶 Auto SLAM (Mobile)</button>
  <button id="btnStationary" class="btn" onclick="setMode('stationary')">📍 Stationary Scan</button>
  <div class="divider"></div>

  <!-- Actions -->
  <button id="btnPause" class="btn" onclick="togglePause()">⏸️ Pause</button>
  <button id="btnReset" class="btn danger" onclick="resetMap()">🔄 Reset Map</button>
  <button id="btnCenter" class="btn" onclick="centerView()">🎯 Center View</button>
  <div class="divider"></div>

  <!-- Measure tool -->
  <button id="btnMeasure" class="btn" onclick="toggleMeasure()">📏 Measure Distance</button>
  <div class="divider"></div>

  <!-- Exports -->
  <button class="btn success" onclick="exportPng()">💾 Save PNG</button>
  <button class="btn success" onclick="exportRos()">🤖 Export ROS Map</button>
</div>

<main id="viewport">
  <canvas id="mapCanvas"></canvas>
  <div class="hud">
    <div>Scroll to <b>Zoom</b> &bull; Drag to <b>Pan</b> &bull; Scale: <b>3cm / cell</b></div>
    <div class="hud-legend">
      <div class="legend-item"><div class="legend-color" style="background: #ffffff; border: 1px solid #666;"></div> Free Space</div>
      <div class="legend-item"><div class="legend-color" style="background: #000000; border: 1px solid #666;"></div> Wall / Obstacle</div>
      <div class="legend-item"><div class="legend-color" style="background: #38bdf8;"></div> Path</div>
      <div class="legend-item"><div class="legend-color" style="background: #10b981;"></div> Live Scan</div>
    </div>
  </div>
  <div class="toast" id="toast"></div>
</main>

<script>
const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');
const viewport = document.getElementById('viewport');
const toastEl = document.getElementById('toast');

// View state
let scale = 2.2;
let panX = 0;
let panY = 0;
let isDragging = false;
let startX = 0, startY = 0;

// Measurement state
let isMeasuring = false;
let measureP1 = null;
let measureP2 = null;

// SLAM data
let mapImg = new Image();
let mapResolution = 0.03; // 3cm per pixel
let mapOriginCells = 300; // 600 / 2
let robotPose = [0, 0, 0];
let trajectory = [];
let livePoints = [];
let isPaused = false;
let currentMode = "slam";

function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  setTimeout(() => toastEl.classList.remove('show'), 2800);
}

function resize() {
  canvas.width = viewport.clientWidth;
  canvas.height = viewport.clientHeight;
  if (panX === 0 && panY === 0) {
    centerView();
  }
}
window.addEventListener('resize', resize);

function centerView() {
  panX = canvas.width / 2;
  panY = canvas.height / 2;
}

// Mouse events for pan & zoom
viewport.addEventListener('mousedown', (e) => {
  if (isMeasuring) {
    const wx = (e.clientX - viewport.getBoundingClientRect().left - panX) / (scale * (1 / mapResolution)) * mapResolution;
    const wy = -(e.clientY - viewport.getBoundingClientRect().top - panY) / (scale * (1 / mapResolution)) * mapResolution;
    if (!measureP1) {
      measureP1 = { x: wx, y: wy };
    } else {
      measureP2 = { x: wx, y: wy };
    }
    return;
  }
  isDragging = true;
  startX = e.clientX - panX;
  startY = e.clientY - panY;
});

viewport.addEventListener('mousemove', (e) => {
  if (isDragging) {
    panX = e.clientX - startX;
    panY = e.clientY - startY;
  }
});

viewport.addEventListener('mouseup', () => { isDragging = false; });
viewport.addEventListener('mouseleave', () => { isDragging = false; });

viewport.addEventListener('wheel', (e) => {
  e.preventDefault();
  const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
  scale = Math.min(Math.max(scale * zoomFactor, 0.4), 8.0);
}, { passive: false });

function toggleMeasure() {
  isMeasuring = !isMeasuring;
  const btn = document.getElementById('btnMeasure');
  btn.classList.toggle('active', isMeasuring);
  viewport.classList.toggle('measuring', isMeasuring);
  if (!isMeasuring) {
    measureP1 = null;
    measureP2 = null;
  } else {
    showToast("Click 2 points on the map to measure distance");
  }
}

// Server API calls
async function setMode(mode) {
  currentMode = mode;
  document.getElementById('btnSlam').classList.toggle('active', mode === 'slam');
  document.getElementById('btnStationary').classList.toggle('active', mode === 'stationary');
  await fetch('/control?action=set_mode&mode=' + mode, { method: 'POST' });
  showToast(mode === 'slam' ? "Auto SLAM mode active" : "Stationary scanning mode active");
}

async function togglePause() {
  isPaused = !isPaused;
  const btn = document.getElementById('btnPause');
  btn.textContent = isPaused ? "▶️ Resume" : "⏸️ Pause";
  btn.classList.toggle('active', isPaused);
  await fetch('/control?action=pause&val=' + isPaused, { method: 'POST' });
}

async function resetMap() {
  if (confirm("Reset current occupancy map and trajectory?")) {
    await fetch('/control?action=reset', { method: 'POST' });
    trajectory = [];
    measureP1 = null;
    measureP2 = null;
    centerView();
    showToast("Map & Trajectory reset!");
  }
}

async function exportPng() {
  const res = await fetch('/control?action=export_png', { method: 'POST' });
  const data = await res.json();
  showToast("Saved: " + data.filename);
}

async function exportRos() {
  const res = await fetch('/control?action=export_ros', { method: 'POST' });
  const data = await res.json();
  showToast("Exported ROS map: " + data.pgm + " & " + data.yaml);
}

// Background poll loops
async function pollState() {
  try {
    const res = await fetch('/state');
    if (res.ok) {
      const data = await res.json();
      robotPose = data.pose;
      trajectory = data.trajectory;
      livePoints = data.live_points;

      document.getElementById('st-fps').textContent = data.stats.fps + ' Hz';
      document.getElementById('st-pose').textContent = data.pose[0].toFixed(2) + 'm, ' + data.pose[1].toFixed(2) + 'm';
      document.getElementById('st-heading').textContent = (data.pose[2] * 180 / Math.PI).toFixed(1) + '°';
      document.getElementById('st-dist').textContent = data.stats.total_dist_m.toFixed(2) + ' m';
      document.getElementById('st-inlier').textContent = Math.round(data.stats.inlier * 100) + '%';
    }
  } catch(e) {}
  setTimeout(pollState, 100);
}

async function pollMapImage() {
  try {
    const img = new Image();
    img.src = '/map_image?t=' + Date.now();
    await img.decode();
    mapImg = img;
  } catch(e) {}
  setTimeout(pollMapImage, 600);
}

// World (meters) to Screen (canvas pixels)
function worldToScreen(wx, wy) {
  const ppm = 1.0 / mapResolution; // pixels per meter on map texture
  const sx = panX + (wx * ppm) * scale;
  const sy = panY - (wy * ppm) * scale;
  return { x: sx, y: sy };
}

// Render Loop
function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const ppm = 1.0 / mapResolution;

  // 1. Draw Occupancy Grid Texture
  if (mapImg && mapImg.complete && mapImg.naturalWidth > 0) {
    const w = mapImg.width * scale;
    const h = mapImg.height * scale;
    const imgX = panX - (mapOriginCells * scale);
    const imgY = panY - (mapOriginCells * scale);

    // Disable image smoothing for sharp pixels
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(mapImg, imgX, imgY, w, h);
  }

  // 2. Draw 1-Meter World Coordinate Grid Lines
  ctx.strokeStyle = 'rgba(71, 85, 105, 0.25)';
  ctx.lineWidth = 1;
  const gridStepM = 1.0;
  const minM = -9, maxM = 9;
  for (let m = minM; m <= maxM; m += gridStepM) {
    const p1 = worldToScreen(m, minM);
    const p2 = worldToScreen(m, maxM);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
    ctx.stroke();

    const p3 = worldToScreen(minM, m);
    const p4 = worldToScreen(maxM, m);
    ctx.beginPath();
    ctx.moveTo(p3.x, p3.y); ctx.lineTo(p4.x, p4.y);
    ctx.stroke();
  }

  // 3. Draw World Axes at (0, 0)
  const o = worldToScreen(0, 0);
  ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
  ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(o.x + 35, o.y); ctx.stroke(); // X (East)
  ctx.strokeStyle = 'rgba(34, 197, 94, 0.6)';
  ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(o.x, o.y - 35); ctx.stroke(); // Y (North)

  // 4. Draw Trajectory Path
  if (trajectory.length > 1) {
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    const start = worldToScreen(trajectory[0][0], trajectory[0][1]);
    ctx.moveTo(start.x, start.y);
    for (let i = 1; i < trajectory.length; i++) {
      const pt = worldToScreen(trajectory[i][0], trajectory[i][1]);
      ctx.lineTo(pt.x, pt.y);
    }
    ctx.stroke();
  }

  // 5. Draw Live Laser Scan Hit Points
  ctx.fillStyle = '#10b981';
  for (const pt of livePoints) {
    const sp = worldToScreen(pt[0], pt[1]);
    ctx.beginPath();
    ctx.arc(sp.x, sp.y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // 6. Draw Robot / Sensor Body & Heading Arrow
  const rPos = worldToScreen(robotPose[0], robotPose[1]);
  const heading = robotPose[2];

  ctx.save();
  ctx.translate(rPos.x, rPos.y);
  ctx.rotate(-heading); // Invert rotation for canvas coords

  // Outer glow ring
  ctx.fillStyle = 'rgba(14, 165, 233, 0.25)';
  ctx.beginPath();
  ctx.arc(0, 0, 14, 0, Math.PI * 2);
  ctx.fill();

  // Robot triangle
  ctx.fillStyle = '#0284c7';
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, -12); // Forward nose
  ctx.lineTo(9, 9);
  ctx.lineTo(0, 4);
  ctx.lineTo(-9, 9);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  // 7. Draw Measurement Line if Active
  if (measureP1) {
    const p1 = worldToScreen(measureP1.x, measureP1.y);
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath(); ctx.arc(p1.x, p1.y, 5, 0, Math.PI * 2); ctx.fill();

    if (measureP2) {
      const p2 = worldToScreen(measureP2.x, measureP2.y);
      ctx.beginPath(); ctx.arc(p2.x, p2.y, 5, 0, Math.PI * 2); ctx.fill();

      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      ctx.setLineDash([]);

      const distM = Math.hypot(measureP2.x - measureP1.x, measureP2.y - measureP1.y);
      const midX = (p1.x + p2.x) / 2;
      const midY = (p1.y + p2.y) / 2 - 8;
      ctx.fillStyle = '#fef3c7';
      ctx.font = 'bold 12px monospace';
      ctx.fillText(`${distM.toFixed(3)} m (${(distM * 100).toFixed(1)} cm)`, midX + 6, midY);
    }
  }

  requestAnimationFrame(render);
}

// Start
resize();
centerView();
pollState();
pollMapImage();
requestAnimationFrame(render);
</script>
</body>
</html>
"""


class MapperHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path.startswith("/map_image"):
            with state_lock:
                rgba = slam.grid.to_rgba()
            img = Image.fromarray(rgba, "RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG", compress_level=1)
            png_bytes = buf.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(png_bytes)

        elif self.path == "/state":
            with state_lock:
                data = {
                    "pose": [round(float(slam.pose[0]), 3), round(float(slam.pose[1]), 3), round(float(slam.pose[2]), 4)],
                    "trajectory": slam.trajectory,
                    "live_points": latest_live_pts,
                    "stats": {
                        "fps": latest_fps,
                        "total_dist_m": round(float(slam.total_distance), 2),
                        "inlier": latest_inlier,
                        "mode": slam.mode,
                    },
                }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global paused
        if self.path.startswith("/control"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            action = query.get("action", [""])[0]

            res_data = {"status": "ok"}
            if action == "reset":
                with state_lock:
                    slam.reset()
                print("[Control] Map and trajectory reset.")

            elif action == "set_mode":
                mode = query.get("mode", ["slam"])[0]
                with state_lock:
                    slam.mode = mode
                print(f"[Control] Mode set to {mode}")

            elif action == "pause":
                val = query.get("val", ["false"])[0].lower() == "true"
                paused = val
                print(f"[Control] Paused: {paused}")

            elif action == "export_png":
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                fn = f"map_{timestamp}.png"
                fp = os.path.join(MAPS_DIR, fn)
                with state_lock:
                    slam.grid.export_png(fp)
                res_data["filename"] = fn
                print(f"[Control] Exported PNG: {fp}")

            elif action == "export_ros":
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base = os.path.join(MAPS_DIR, f"map_{timestamp}")
                with state_lock:
                    pgm, yml = slam.grid.export_ros_map(base)
                res_data["pgm"] = os.path.basename(pgm)
                res_data["yaml"] = os.path.basename(yml)
                print(f"[Control] Exported ROS Map: {pgm}, {yml}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res_data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return  # Suppress request spam in console


def main():
    parser = argparse.ArgumentParser(description="LiDAR 2D SLAM & Room Mapper")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    args = parser.parse_args()

    # Start LiDAR SLAM worker thread
    t = threading.Thread(target=lidar_slam_thread, daemon=True)
    t.start()

    port = args.port
    server = None
    # Try preferred port, or fall back to port + 1 if busy
    for p in [port, 8082, 8085]:
        try:
            server = HTTPServer(("0.0.0.0", p), MapperHandler)
            port = p
            break
        except OSError:
            continue

    if server is None:
        print(f"Error: Could not bind to port {args.port} or alternatives.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🗺️  LiDAR 2D SLAM & Floor Plan Mapper is RUNNING!")
    print(f"👉 Open in your browser: http://localhost:{port}")
    print(f"📁 Exported maps will be saved to: {MAPS_DIR}")
    print("=" * 60 + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping LiDAR Mapper...")
        global running
        running = False
        server.server_close()


if __name__ == "__main__":
    main()
