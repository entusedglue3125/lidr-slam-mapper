# 2D LiDAR SLAM & Room Mapper

An open-source, real-time 2D SLAM (Simultaneous Localization and Mapping) and Occupancy Grid Floor Plan generator for YDLIDAR / 2D laser scanners.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Linux-green)
![ROS Compatible](https://img.shields.io/badge/ROS%20%2F%20ROS2-Nav2%20Compatible-orange)

---

## Features

* **Real-Time 2D SLAM**: Sub-pixel ICP (Iterative Closest Point) scan matcher tracks position $(X, Y)$ and orientation $\Theta$ without needing wheel encoders or IMUs.
* **Log-Odds Occupancy Grid**: Ray-traces free space and obstacles using Bresenham lines at 3cm/cell resolution across an 18m $\times$ 18m canvas.
* **Interactive Web Dashboard**:
  * Real-time map rendering with pan & zoom controls.
  * Live laser scan overlay (green dots) and historical robot trajectory path (cyan line).
  * **Digital Ruler Tool**: Click any 2 points on the room map to measure exact real-world distances in meters and centimeters.
* **Modes**:
  * **Auto SLAM (Mobile)**: Walk around with the LiDAR or mount it on a robot to map multiple rooms.
  * **Stationary Scan**: High-density room scanning from a single fixed spot to eliminate drift.
* **Export Options**:
  * **High-Res PNG**: Instantly save clear floor plan pictures.
  * **ROS / ROS2 Maps**: Exports standard `.pgm` and `.yaml` map files ready to load into `rviz`, `Nav2`, or `slam_toolbox`.

---

## Quick Start

### 1. Launch the 2D SLAM Mapper
> **Note**: Make sure other scripts accessing `/dev/ttyUSB0` are stopped first.

```bash
cd /home/chakradhar/Downloads/LiDr
./venv/bin/python3 mapper_app.py
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser.

### 2. Available Scripts

| Script | Purpose |
| :--- | :--- |
| [`mapper_app.py`](file:///home/chakradhar/Downloads/LiDr/mapper_app.py) | **Full 2D SLAM & Floor Plan Mapper** with web UI. |
| [`web_viewer.py`](file:///home/chakradhar/Downloads/LiDr/web_viewer.py) | Simple 360° circular sweep radar display. |
| [`terminal_radar.py`](file:///home/chakradhar/Downloads/LiDr/terminal_radar.py) | ASCII top-down radar inside the terminal. |
| [`ydlidar.py`](file:///home/chakradhar/Downloads/LiDr/ydlidar.py) | Core Python driver for reading raw LiDAR scans. |
| [`mapper.py`](file:///home/chakradhar/Downloads/LiDr/mapper.py) | Standalone ICP scan matcher & occupancy grid engine. |

---

## Saved Maps

All exported maps are stored in the [`maps/`](file:///home/chakradhar/Downloads/LiDr/maps) directory:
* `map_<timestamp>.png` — Visual floor plan
* `map_<timestamp>.pgm` — ROS occupancy grid image
* `map_<timestamp>.yaml` — ROS map metadata (resolution, origin)
