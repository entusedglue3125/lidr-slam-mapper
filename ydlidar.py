"""
YDLIDAR Driver for Python
Compatible with YDLIDAR models using standard protocol at 115200 baud (e.g. X2, X2L, X3, etc.).
"""

import time
import math
import serial


class YDLidar:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def connect(self):
        if self.ser is None or not self.ser.is_open:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self.ser.reset_input_buffer()
        return self

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def _read_packet(self):
        """Find next packet header and parse packet."""
        while True:
            b1 = self.ser.read(1)
            if not b1:
                return None, []
            if b1 == b"\xaa":
                b2 = self.ser.read(1)
                if b2 == b"\x55":
                    header = self.ser.read(8)
                    if len(header) < 8:
                        continue
                    ct = header[0]
                    lsn = header[1]
                    fsa = header[2] | (header[3] << 8)
                    lsa = header[4] | (header[5] << 8)
                    cs = header[6] | (header[7] << 8)

                    start_deg = (fsa >> 1) / 64.0
                    end_deg = (lsa >> 1) / 64.0

                    payload = self.ser.read(lsn * 2)
                    if len(payload) < lsn * 2:
                        continue

                    points = []
                    diff = end_deg - start_deg
                    if diff < 0:
                        diff += 360.0

                    for i in range(lsn):
                        raw_dist = payload[2 * i] | (payload[2 * i + 1] << 8)
                        dist_mm = raw_dist / 4.0
                        if lsn > 1:
                            ang = (start_deg + (diff / (lsn - 1)) * i) % 360.0
                        else:
                            ang = start_deg
                        points.append((ang, dist_mm))

                    return ct, points

    def iter_scans(self, filter_zero=True):
        """
        Yields full 360-degree scans as lists of (angle_deg, distance_mm).
        A scan starts when CT & 0x01 == 1 (zero angle packet).
        """
        current_scan = []
        synced = False

        while True:
            ct, points = self._read_packet()
            if ct is None:
                continue

            is_zero_packet = bool(ct & 1)

            if is_zero_packet:
                if synced and current_scan:
                    if filter_zero:
                        yield [(a, d) for a, d in current_scan if d > 0]
                    else:
                        yield current_scan
                    current_scan = []
                synced = True

            if synced:
                current_scan.extend(points)


if __name__ == "__main__":
    print("Testing YDLidar driver on /dev/ttyUSB0...")
    with YDLidar() as lidar:
        count = 0
        for scan in lidar.iter_scans():
            count += 1
            if scan:
                closest = min(scan, key=lambda p: p[1])
                furthest = max(scan, key=lambda p: p[1])
                avg_dist = sum(p[1] for p in scan) / len(scan)
                print(
                    f"Scan #{count:02d}: {len(scan)} points | "
                    f"Closest: {closest[1]:.0f}mm @ {closest[0]:.1f}° | "
                    f"Furthest: {furthest[1]:.0f}mm @ {furthest[0]:.1f}° | "
                    f"Avg: {avg_dist:.0f}mm"
                )
            if count >= 5:
                break
    print("Test completed successfully!")
