"""
2D LiDAR SLAM & Occupancy Grid Mapping Engine.
Features:
- Sub-pixel 2D ICP scan matching (odometry tracking)
- Log-odds occupancy grid mapping with Bresenham ray-tracing
- Trajectory tracking
- Export to PNG and standard ROS (PGM + YAML) formats
"""

import math
import time
import numpy as np
from PIL import Image


def bresenham_line(x0, y0, x1, y1):
    """Generates (x, y) coordinates along a line using Bresenham algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    points = []
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points


class OccupancyGrid:
    def __init__(self, size_cells=500, resolution_m=0.03):
        """
        size_cells: width & height in grid cells (default 500x500)
        resolution_m: meters per grid cell (default 0.03m = 3cm per pixel)
        Total physical area: size_cells * resolution_m (default 15m x 15m)
        """
        self.size = size_cells
        self.resolution = resolution_m
        self.origin_x = size_cells // 2
        self.origin_y = size_cells // 2

        # Log-odds grid: 0 = unknown (P=0.5), positive = occupied, negative = free
        self.log_odds = np.zeros((size_cells, size_cells), dtype=np.float32)

        # Log-odds update values
        self.l_free = -0.35
        self.l_occ = 0.85
        self.l_min = -3.5
        self.l_max = 3.5

    def reset(self):
        self.log_odds.fill(0)

    def world_to_grid(self, wx, wy):
        """Convert world coordinates (meters) to grid coordinates (pixels)."""
        gx = int(np.round(self.origin_x + wx / self.resolution))
        gy = int(np.round(self.origin_y - wy / self.resolution))  # Invert Y for screen coords
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Convert grid coordinates (pixels) to world coordinates (meters)."""
        wx = (gx - self.origin_x) * self.resolution
        wy = (self.origin_y - gy) * self.resolution
        return wx, wy

    def update_scan(self, sensor_x, sensor_y, hit_points_world, clear_free=True):
        """
        Updates the occupancy grid given sensor position and world-frame hit points.
        hit_points_world: Nx2 array of [x, y] in meters.
        """
        sgx, sgy = self.world_to_grid(sensor_x, sensor_y)
        if not (0 <= sgx < self.size and 0 <= sgy < self.size):
            return

        for wx, wy in hit_points_world:
            hgx, hgy = self.world_to_grid(wx, wy)

            if not (0 <= hgx < self.size and 0 <= hgy < self.size):
                continue

            if clear_free:
                # Ray trace from sensor to obstacle
                line_pts = bresenham_line(sgx, sgy, hgx, hgy)
                # Free space along the ray (excluding obstacle cell)
                for fx, fy in line_pts[:-1]:
                    if 0 <= fx < self.size and 0 <= fy < self.size:
                        self.log_odds[fy, fx] = max(self.l_min, self.log_odds[fy, fx] + self.l_free)

            # Mark occupied at endpoint
            self.log_odds[hgy, hgx] = min(self.l_max, self.log_odds[hgy, hgx] + self.l_occ)

    def to_rgba(self):
        """
        Convert log-odds grid to RGBA uint8 image array:
        - Unknown (log-odds ~ 0): neutral gray (128, 128, 128)
        - Free (log-odds < 0): light white/cream (250, 250, 250)
        - Occupied (log-odds > 0): dark slate/black (20, 20, 25)
        """
        # Convert log-odds to probability P in [0, 1]
        p = 1.0 / (1.0 + np.exp(-self.log_odds))

        # Map to grayscale values (0 = occupied/black, 255 = free/white, 128 = unknown)
        gray = np.clip((1.0 - p) * 255.0, 0, 255).astype(np.uint8)

        rgba = np.zeros((self.size, self.size, 4), dtype=np.uint8)
        rgba[..., 0] = gray
        rgba[..., 1] = gray
        rgba[..., 2] = gray
        rgba[..., 3] = 255
        return rgba

    def export_png(self, filepath):
        """Exports the occupancy grid as a PNG image."""
        rgba = self.to_rgba()
        img = Image.fromarray(rgba, "RGBA")
        img.save(filepath)
        return filepath

    def export_ros_map(self, base_name="map"):
        """
        Exports standard ROS Map (base_name.pgm and base_name.yaml).
        Usable directly by ROS / ROS2 nav2 and slam_toolbox.
        """
        p = 1.0 / (1.0 + np.exp(-self.log_odds))
        # ROS convention: 0 = free (white), 100 = occupied (black), 205 = unknown
        ros_grid = np.zeros_like(p, dtype=np.uint8)
        ros_grid[np.abs(self.log_odds) < 0.2] = 205  # Unknown
        ros_grid[self.log_odds <= -0.2] = 254        # Free
        ros_grid[self.log_odds >= 0.2] = 0           # Occupied

        pgm_path = f"{base_name}.pgm"
        yaml_path = f"{base_name}.yaml"

        img = Image.fromarray(ros_grid, "L")
        img.save(pgm_path)

        # Origin in world coordinates: bottom-left corner
        min_x = -self.origin_x * self.resolution
        min_y = -(self.size - self.origin_y) * self.resolution

        yaml_content = (
            f"image: {base_name}.pgm\n"
            f"resolution: {self.resolution:.4f}\n"
            f"origin: [{min_x:.4f}, {min_y:.4f}, 0.0]\n"
            f"negate: 0\n"
            f"occupied_thresh: 0.65\n"
            f"free_thresh: 0.196\n"
        )
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        return pgm_path, yaml_path


class ScanMatcher2D:
    """Fast 2D ICP Scan Matcher for LiDAR odometry estimation."""

    def __init__(self, max_iter=20, max_dist=0.35, tol=1e-4):
        self.max_iter = max_iter
        self.max_dist = max_dist
        self.tol = tol

    def match(self, current_pts, previous_pts):
        """
        Aligns current_pts to previous_pts.
        Returns: dx, dy (meters), dtheta (radians), inlier_ratio
        """
        if len(current_pts) < 15 or len(previous_pts) < 15:
            return 0.0, 0.0, 0.0, 0.0

        p = current_pts.copy()
        theta_acc = 0.0
        t_acc = np.zeros(2)

        for _ in range(self.max_iter):
            # Compute pair-wise distances
            diff = p[:, np.newaxis, :] - previous_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))

            nearest_idx = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(len(p)), nearest_idx]

            inliers = min_dists < self.max_dist
            if np.sum(inliers) < 15:
                break

            p_in = p[inliers]
            q_in = previous_pts[nearest_idx[inliers]]

            p_mean = np.mean(p_in, axis=0)
            q_mean = np.mean(q_in, axis=0)

            H = (p_in - p_mean).T @ (q_in - q_mean)
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            d_th = math.atan2(R[1, 0], R[0, 0])
            dt = q_mean - (R @ p_mean)

            # Update working points
            p = (R @ p.T).T + dt

            # Update accumulated transform
            R_step = np.array([
                [math.cos(d_th), -math.sin(d_th)],
                [math.sin(d_th), math.cos(d_th)]
            ])
            t_acc = (R_step @ t_acc) + dt
            theta_acc += d_th

            if np.linalg.norm(dt) < self.tol and abs(d_th) < self.tol:
                break

        inlier_ratio = np.sum(inliers) / len(current_pts) if len(current_pts) > 0 else 0.0
        return t_acc[0], t_acc[1], theta_acc, inlier_ratio


class LidarSLAM:
    """Integrates scan matching and occupancy grid mapping."""

    def __init__(self, size_cells=500, resolution_m=0.03):
        self.grid = OccupancyGrid(size_cells, resolution_m)
        self.matcher = ScanMatcher2D()

        # Robot Pose: [x, y, theta]
        self.pose = np.array([0.0, 0.0, 0.0])  # x (m), y (m), theta (rad)
        self.trajectory = [[0.0, 0.0]]
        self.total_distance = 0.0

        self.prev_scan_pts = None
        self.mode = "slam"  # "slam" (auto-tracking) or "stationary"
        self.clear_free = True

    def reset(self):
        self.grid.reset()
        self.pose = np.array([0.0, 0.0, 0.0])
        self.trajectory = [[0.0, 0.0]]
        self.total_distance = 0.0
        self.prev_scan_pts = None

    def process_scan(self, scan):
        """
        Processes one full 360-degree scan (list of (deg, mm)).
        Updates pose, trajectory, and occupancy grid.
        Returns:
            world_pts: Nx2 array of points in world coordinates
            inlier_ratio: Scan match quality (0.0 to 1.0)
        """
        if not scan:
            return np.empty((0, 2)), 0.0

        # Convert polar to local Cartesian (meters)
        # 0 deg = North (Y+), 90 deg = East (X+)
        angles = np.array([math.radians(p[0]) for p in scan])
        dists = np.array([p[1] / 1000.0 for p in scan])

        # Filter out extreme outliers / zero readings
        valid = (dists > 0.08) & (dists < 10.0)
        if np.sum(valid) < 10:
            return np.empty((0, 2)), 0.0

        local_x = dists[valid] * np.sin(angles[valid])
        local_y = dists[valid] * np.cos(angles[valid])
        local_pts = np.column_stack([local_x, local_y])

        inlier_ratio = 1.0
        # If SLAM mode is active, track motion using scan matching
        if self.mode == "slam" and self.prev_scan_pts is not None:
            dx, dy, dth, inlier_ratio = self.matcher.match(local_pts, self.prev_scan_pts)

            # Sanity check: discard insane jumps (e.g. >0.4m or >45 deg per 100ms)
            if math.hypot(dx, dy) < 0.4 and abs(dth) < math.radians(45):
                # Transform delta to world frame using current heading
                c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
                world_dx = c * dx - s * dy
                world_dy = s * dx + c * dy

                self.pose[0] += world_dx
                self.pose[1] += world_dy
                self.pose[2] = (self.pose[2] + dth) % (2 * math.pi)

                step_dist = math.hypot(world_dx, world_dy)
                self.total_distance += step_dist

                # Record trajectory point every 5cm of travel or 5 deg rotation
                last_traj = self.trajectory[-1]
                if math.hypot(self.pose[0] - last_traj[0], self.pose[1] - last_traj[1]) > 0.05:
                    self.trajectory.append([round(self.pose[0], 3), round(self.pose[1], 3)])

        self.prev_scan_pts = local_pts

        # Transform local points to world coordinates using current pose
        th = self.pose[2]
        c, s = math.cos(th), math.sin(th)
        R_world = np.array([[c, -s], [s, c]])

        world_pts = (R_world @ local_pts.T).T + self.pose[:2]

        # Update occupancy grid map
        self.grid.update_scan(self.pose[0], self.pose[1], world_pts, clear_free=self.clear_free)

        return world_pts, inlier_ratio
