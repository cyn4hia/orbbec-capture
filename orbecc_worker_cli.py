"""
Headless Orbbec capture worker for the GUI. Mac/Linux only.

Invocation: python orbbec_worker_cli.py <json_payload>
Output: one line "CAPTURE_RESULT:{...json...}"

Saves point cloud to <out_dir>/orbbec.ply
"""

from __future__ import annotations

import json
import struct
import sys
import traceback
from datetime import datetime
from pathlib import Path

_PYORBBECSDK_LIB = "/Users/cyn/coding/projects/pyorbbecsdk/install/lib"
if _PYORBBECSDK_LIB not in sys.path:
    sys.path.insert(0, _PYORBBECSDK_LIB)

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def capture_orbbec(out_dir: Path) -> dict:
    try:
        from pyorbbecsdk import (
            Pipeline,
            Config,
            OBSensorType,
            OBStreamType,
            OBFormat,
            FrameSet,
            PointCloudFilter,
            OBAlignMode,
        )
    except Exception as e:
        return {"status": "failed", "error": f"pyorbbecsdk import failed: {e}"}

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        ply_path = out_dir / "orbbec.ply"

        pipeline = Pipeline()
        config = Config()

        # Try to enable depth + color streams. We pick a sensible default
        # profile -- if the camera doesn't support it, fall back to default.
        try:
            depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
        except Exception as e:
            return {"status": "failed", "error": f"depth stream not available: {e}"}

        try:
            color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            has_color = True
        except Exception:
            has_color = False  # color not strictly required

        # Align depth to color so the point cloud is well-formed
        try:
            config.set_align_mode(OBAlignMode.HW_MODE)
        except Exception:
            pass

        pipeline.start(config)

        # Discard a few warm-up frames -- exposure stabilizes on real cameras
        for _ in range(5):
            pipeline.wait_for_frames(1000)

        frames: FrameSet = pipeline.wait_for_frames(2000)
        if frames is None:
            pipeline.stop()
            return {"status": "failed", "error": "no frames received from Orbbec"}

        # Build point cloud
        pc_filter = PointCloudFilter()
        try:
            pc_filter.set_create_point_format(
                OBFormat.RGB_POINT if has_color else OBFormat.POINT
            )
            camera_param = pipeline.get_camera_param()
            pc_filter.set_camera_param(camera_param)
        except Exception:
            pass

        point_frame = pc_filter.process(frames)
        if point_frame is None:
            pipeline.stop()
            return {"status": "failed", "error": "point cloud filter returned no frame"}

        # Get a numpy array of points -- pyorbbecsdk exposes get_point_cloud_data()
        # but field names vary by version, so we fall through.
        points = None
        for getter in ("get_point_cloud_data", "get_data"):
            if hasattr(point_frame, getter):
                try:
                    points = getattr(point_frame, getter)()
                    break
                except Exception:
                    pass
        if points is None:
            pipeline.stop()
            return {"status": "failed",
                    "error": "could not extract point data from frame"}

        import numpy as np
        arr = np.asarray(points)

        # arr layout depends on whether color is present:
        #   RGB_POINT -> (N, 6) [x, y, z, r, g, b]
        #   POINT     -> (N, 3) [x, y, z]
        if arr.ndim == 1:
            cols = 6 if has_color else 3
            arr = arr.reshape(-1, cols)

        n = arr.shape[0]
        with_color = has_color and arr.shape[1] >= 6

        # Write binary PLY
        if with_color:
            header = (
                f"ply\nformat binary_little_endian 1.0\n"
                f"element vertex {n}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                f"end_header\n"
            ).encode("ascii")
            with ply_path.open("wb") as f:
                f.write(header)
                xyz = arr[:, 0:3].astype(np.float32)
                rgb = arr[:, 3:6].astype(np.uint8)
                # Interleave xyz (12 bytes) + rgb (3 bytes) per point
                for i in range(n):
                    f.write(xyz[i].tobytes())
                    f.write(rgb[i].tobytes())
        else:
            header = (
                f"ply\nformat binary_little_endian 1.0\n"
                f"element vertex {n}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"end_header\n"
            ).encode("ascii")
            with ply_path.open("wb") as f:
                f.write(header)
                f.write(arr[:, 0:3].astype(np.float32).tobytes())

        pipeline.stop()
        size = ply_path.stat().st_size if ply_path.exists() else 0

        return {
            "status": "complete",
            "file": str(ply_path),
            "size_bytes": size,
            "settings": {
                "color": with_color,
                "n_points": int(n),
            },
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def main() -> int:
    if len(sys.argv) < 2:
        print("CAPTURE_RESULT:" + json.dumps({
            "status": "failed", "error": "no payload"}))
        return 1

    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print("CAPTURE_RESULT:" + json.dumps({
            "status": "failed", "error": f"bad json: {e}"}))
        return 1

    cameras = [c.lower() for c in payload.get("cameras", [])]
    out_dir = Path(payload["out_dir"])

    files: dict[str, str] = {}
    settings: dict[str, dict] = {}
    errors: list[str] = []

    if "orbbec" in cameras:
        result = capture_orbbec(out_dir)
        if result["status"] == "complete":
            files["orbbec"] = result["file"]
            settings["orbbec"] = result.get("settings", {})
        else:
            errors.append(f"orbbec: {result.get('error', 'unknown')}")

    if files and not errors:
        overall = "complete"
    elif files:
        overall = "partial"
    else:
        overall = "failed"

    out: dict = {
        "status": overall,
        "files": files,
        "camera_settings": settings,
        "timestamp": _now_iso(),
    }
    if errors:
        out["error"] = "; ".join(errors)

    print("CAPTURE_RESULT:" + json.dumps(out))
    return 0 if overall != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())