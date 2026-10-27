"""
Interactive Web Visualizer for YDLIDAR.
Hosts a local web server (http://localhost:8080) with a live radar view on HTML5 Canvas.
Zero external dependencies beyond pyserial.
"""

import json
import math
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from ydlidar import YDLidar

# Global scan buffer protected by lock
scan_lock = threading.Lock()
latest_scan_data = {
    "points": [],
    "stats": {
        "count": 0,
        "closest_mm": 0,
        "closest_deg": 0,
        "fps": 0.0,
    },
}
running = True


def lidar_worker():
    global latest_scan_data, running
    fps_counter = 0
    t0 = time.time()
    current_fps = 0.0

    while running:
        try:
            with YDLidar() as lidar:
                for scan in lidar.iter_scans():
                    if not running:
                        break
                    fps_counter += 1
                    now = time.time()
                    if now - t0 >= 1.0:
                        current_fps = round(fps_counter / (now - t0), 1)
                        fps_counter = 0
                        t0 = now

                    # Format points
                    pts = []
                    closest_pt = min(scan, key=lambda p: p[1]) if scan else (0, 0)
                    for ang_deg, dist_mm in scan:
                        rad = math.radians(ang_deg)
                        # x: right, y: forward (up)
                        x = (dist_mm / 1000.0) * math.sin(rad)
                        y = (dist_mm / 1000.0) * math.cos(rad)
                        pts.append([round(x, 3), round(y, 3), round(dist_mm, 1), round(ang_deg, 1)])

                    with scan_lock:
                        latest_scan_data = {
                            "points": pts,
                            "stats": {
                                "count": len(pts),
                                "closest_mm": round(closest_pt[1], 1),
                                "closest_deg": round(closest_pt[0], 1),
                                "fps": current_fps,
                            },
                        }
        except Exception as e:
            print(f"[LiDAR Worker] Reconnecting after error: {e}")
            time.sleep(1.0)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>YDLIDAR Real-Time Radar</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
  body { background: #0b0f19; color: #e2e8f0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  header { background: #111827; padding: 12px 24px; border-bottom: 1px solid #1f2937; display: flex; align-items: center; justify-content: space-between; }
  h1 { font-size: 1.15rem; font-weight: 600; color: #10b981; display: flex; align-items: center; gap: 8px; }
  .pulse { width: 10px; height: 10px; background: #10b981; border-radius: 50%; box-shadow: 0 0 10px #10b981; animation: blink 1.4s infinite; }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  .stats-bar { display: flex; gap: 20px; font-size: 0.9rem; }
  .stat-card { background: #1f2937; padding: 4px 12px; border-radius: 6px; border: 1px solid #374151; }
  .stat-label { color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; }
  .stat-val { font-weight: 700; color: #f3f4f6; }
  
  .controls { display: flex; gap: 18px; align-items: center; background: #111827; padding: 8px 24px; border-bottom: 1px solid #1f2937; font-size: 0.85rem; }
  .control-group { display: flex; align-items: center; gap: 8px; }
  input[type="range"] { accent-color: #10b981; }

  main { flex: 1; display: flex; justify-content: center; align-items: center; position: relative; }
  canvas { background: radial-gradient(circle at center, #111e2e 0%, #080d16 100%); border-radius: 50%; box-shadow: 0 0 40px rgba(16, 185, 129, 0.15); border: 2px solid #1f2937; }
  .tooltip { position: absolute; pointer-events: none; background: rgba(17, 24, 39, 0.95); padding: 6px 10px; border-radius: 4px; border: 1px solid #10b981; font-size: 0.8rem; display: none; }
</style>
</head>
<body>

<header>
  <h1><div class="pulse"></div> YDLIDAR Active Scanner</h1>
  <div class="stats-bar">
    <div class="stat-card"><div class="stat-label">Refresh Rate</div><div class="stat-val" id="stat-fps">-- Hz</div></div>
    <div class="stat-card"><div class="stat-label">Points / Rev</div><div class="stat-val" id="stat-count">--</div></div>
    <div class="stat-card"><div class="stat-label">Closest Obstacle</div><div class="stat-val" id="stat-closest">-- mm</div></div>
  </div>
</header>

<div class="controls">
  <div class="control-group">
    <label for="rangeSlider">Max Range: <b id="rangeVal">3.0m</b></label>
    <input id="rangeSlider" type="range" min="1" max="10" step="0.5" value="3.0">
  </div>
  <div class="control-group">
    <label for="pointSize">Point Size: <b id="sizeVal">3px</b></label>
    <input id="pointSize" type="range" min="1" max="6" step="1" value="3">
  </div>
  <div class="control-group">
    <label><input type="checkbox" id="showRings" checked> Range Rings</label>
  </div>
</div>

<main id="container">
  <canvas id="radarCanvas"></canvas>
  <div class="tooltip" id="tooltip"></div>
</main>

<script>
const canvas = document.getElementById('radarCanvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('container');
const rangeSlider = document.getElementById('rangeSlider');
const rangeVal = document.getElementById('rangeVal');
const pointSizeInput = document.getElementById('pointSize');
const sizeVal = document.getElementById('sizeVal');
const showRingsInput = document.getElementById('showRings');
const tooltip = document.getElementById('tooltip');

let maxRangeM = parseFloat(rangeSlider.value);
let pointSize = parseInt(pointSizeInput.value);
let currentPoints = [];
let sweepAngle = 0;

rangeSlider.oninput = (e) => {
  maxRangeM = parseFloat(e.target.value);
  rangeVal.textContent = maxRangeM.toFixed(1) + 'm';
};
pointSizeInput.oninput = (e) => {
  pointSize = parseInt(e.target.value);
  sizeVal.textContent = pointSize + 'px';
};

function resize() {
  const size = Math.min(container.clientWidth - 40, container.clientHeight - 40);
  canvas.width = size;
  canvas.height = size;
}
window.addEventListener('resize', resize);
resize();

// Fetch loop
async function fetchScan() {
  try {
    const res = await fetch('/scan');
    if (res.ok) {
      const data = await res.json();
      currentPoints = data.points;
      document.getElementById('stat-fps').textContent = data.stats.fps + ' Hz';
      document.getElementById('stat-count').textContent = data.stats.count;
      document.getElementById('stat-closest').textContent = data.stats.closest_mm + ' mm @ ' + data.stats.closest_deg + '°';
    }
  } catch(e) {}
  setTimeout(fetchScan, 100);
}
fetchScan();

// Draw loop
function draw() {
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const radius = w / 2 - 10;

  ctx.clearRect(0, 0, w, h);

  // Range rings & degree lines
  if (showRingsInput.checked) {
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#64748b';
    ctx.font = '10px monospace';

    // Crosshairs
    ctx.beginPath();
    ctx.moveTo(cx, 10); ctx.lineTo(cx, h - 10);
    ctx.moveTo(10, cy); ctx.lineTo(w - 10, cy);
    ctx.stroke();

    // Rings every 0.5m or 1.0m depending on range
    const step = maxRangeM <= 4 ? 0.5 : 1.0;
    for (let r = step; r <= maxRangeM; r += step) {
      const pxDist = (r / maxRangeM) * radius;
      ctx.beginPath();
      ctx.arc(cx, cy, pxDist, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillText(r.toFixed(1) + 'm', cx + 4, cy - pxDist - 2);
    }
  }

  // Draw points
  for (const pt of currentPoints) {
    const [x, y, dist_mm, ang_deg] = pt;
    const dist_m = dist_mm / 1000.0;
    if (dist_m <= maxRangeM) {
      const px = cx + (x / maxRangeM) * radius;
      const py = cy - (y / maxRangeM) * radius; // inverted y for screen forward

      // Color coding based on proximity (red = close, cyan/green = far)
      const ratio = Math.min(1.0, dist_m / maxRangeM);
      const r = Math.floor(255 * (1 - ratio));
      const g = Math.floor(180 * ratio + 80);
      const b = Math.floor(255 * ratio);

      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.beginPath();
      ctx.arc(px, py, pointSize, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Draw LiDAR Center
  ctx.fillStyle = '#10b981';
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.fill();

  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
</script>
</body>
</html>
"""


class RadarHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/scan":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with scan_lock:
                data_bytes = json.dumps(latest_scan_data).encode("utf-8")
            self.wfile.write(data_bytes)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging
        return


def run_server(port=8080):
    global running
    # Start LiDAR thread
    t = threading.Thread(target=lidar_worker, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", port), RadarHandler)
    print(f"\n=======================================================")
    print(f"🚀 YDLIDAR Web Radar is LIVE!")
    print(f"👉 Open in your browser: http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server...")
        running = False
        server.server_close()


if __name__ == "__main__":
    run_server(8080)
