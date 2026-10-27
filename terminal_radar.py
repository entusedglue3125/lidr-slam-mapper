"""
Live Terminal ASCII Radar for YDLIDAR.
Renders a real-time top-down 2D view of the surroundings directly in the terminal.
"""

import os
import sys
import math
import time
from ydlidar import YDLidar

WIDTH = 61
HEIGHT = 31
MAX_RANGE_M = 3.0  # Max distance shown in radar in meters


def render_radar(scan, width=WIDTH, height=HEIGHT, max_range=MAX_RANGE_M):
    # Initialize grid
    grid = [[" " for _ in range(width)] for _ in range(height)]
    cx = width // 2
    cy = height // 2

    # Draw range rings (1m, 2m, 3m)
    for r_m in [1.0, 2.0, 3.0]:
        if r_m > max_range:
            continue
        rx = int(r_m / max_range * (cx - 1))
        ry = int(r_m / max_range * (cy - 1))
        for deg in range(0, 360, 5):
            rad = math.radians(deg)
            px = int(cx + rx * math.cos(rad))
            py = int(cy + ry * math.sin(rad))
            if 0 <= px < width and 0 <= py < height:
                grid[py][px] = "·"

    # Draw axis crosshairs
    for x in range(width):
        if grid[cy][x] == " ":
            grid[cy][x] = "-"
    for y in range(height):
        if grid[y][cx] == " " or grid[y][cx] == "-":
            grid[y][cx] = "|"
    grid[cy][cx] = "+"

    # Plot obstacle points
    for angle_deg, dist_mm in scan:
        dist_m = dist_mm / 1000.0
        if 0.05 < dist_m <= max_range:
            rad = math.radians(angle_deg)
            # Standard orientation: 0 deg = North/Forward (up), 90 deg = East/Right
            # In screen coords: y is down, x is right
            # Forward: -cos(rad) for y, sin(rad) for x
            px = int(cx + (dist_m / max_range) * (cx - 1) * math.sin(rad))
            py = int(cy - (dist_m / max_range) * (cy - 1) * math.cos(rad))
            if 0 <= px < width and 0 <= py < height:
                grid[py][px] = "#"

    lines = ["".join(row) for row in grid]
    return "\n".join(lines)


def main():
    print("\033[2J\033[H", end="")  # Clear screen
    print("Connecting to YDLIDAR on /dev/ttyUSB0... (Press Ctrl+C to stop)")
    with YDLidar() as lidar:
        scan_idx = 0
        t0 = time.time()
        for scan in lidar.iter_scans():
            scan_idx += 1
            fps = scan_idx / max(0.001, time.time() - t0)
            closest = min(scan, key=lambda p: p[1]) if scan else (0, 0)
            radar_str = render_radar(scan)

            output = (
                "\033[H"
                f"=== YDLIDAR Real-Time Radar [Max Range: {MAX_RANGE_M:.1f}m] ===\n"
                f"Points: {len(scan):3d} | Rate: {fps:4.1f} Hz | "
                f"Closest: {closest[1]:.0f}mm @ {closest[0]:.1f}°\n"
                f"{radar_str}\n"
                "Legend: [+] LiDAR Center  [·] Range rings (1m, 2m, 3m)  [#] Obstacle\n"
            )
            sys.stdout.write(output)
            sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nRadar stopped.")
