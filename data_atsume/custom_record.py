"""Minimal dataset episode creator and proprio recorder.

Creates a Custom Dataset v1 layout under a dataset root as described in
data-format-v1.md, then records synthetic proprioceptive states to
state.parquet. Designed as a simple prototype with minimal deps.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List
from zoneinfo import ZoneInfo
import numpy as np
import cv2


@dataclass
class ProprioSpec:
    """Specification for synthetic robot proprioception."""

    n_joints: int = 6
    hz: float = 50.0
    duration_s: float = 2.0


def iso_utc_now() -> str:
    """Return e.g. 2025-08-18T17-20-33 using Asia/Singapore timezone."""
    now = datetime.now(ZoneInfo("Asia/Singapore"))
    return now.strftime("%Y-%m-%dT%H-%M-%S")


def init_episode(root: Path, cams: List[str]) -> Path:
    """Create a new episode directory with per-camera subfolders and files."""
    ep_dir = root / f"episode_{iso_utc_now()}"
    for cam in cams:
        (ep_dir / "cams" / cam / "rgb").mkdir(parents=True, exist_ok=True)
        (ep_dir / "cams" / cam / "depth").mkdir(parents=True, exist_ok=True)

    # Calibration (single file at episode root) minimal stub per camera
    calib = {
        "version": 1,
        "frames": {"base_link": {}, "ee_link": {}},
        "cameras": {},
    }
    for cam in cams:
        calib["cameras"][cam] = {
            "model": "pinhole",
            "resolution": [848, 480],
            "intrinsics": {},
            "distortion": {},
            "depth_unit": "mm",
            "depth_scale": 0.001,
            "parent_frame": "base_link",
            "T_parent_camera": {
                "translation_m": [0.0, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
    (ep_dir / "calibration.yaml").write_text(_to_yaml(calib))

    # Metadata with episode_id and camera list
    episode_id = ep_dir.name.replace("episode_", "")
    metadata = {
        "episode_id": episode_id,
        "created_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "units": {"length": "m", "angle": "rad", "time": "ns"},
        "task_name": "demo",
        "nl_prompt": "",
        "primary_clock": "proprio",
        "sync_strategy": "nearest",
        "max_skew_ns": 50_000_000,
        "cameras": cams,
        "robot_model": "unknown",
        "action_convention": "joint_delta",
    }
    (ep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Create empty global index.jsonl
    (ep_dir / "index.jsonl").write_text("")
    return ep_dir


def _to_yaml(d: Dict) -> str:
    """Very small YAML emitter to avoid adding dependencies."""

    def emit(obj, indent=0):
        pad = "  " * indent
        if isinstance(obj, dict):
            lines = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}{k}:")
                    lines.append(emit(v, indent + 1))
                else:
                    sval = "null" if v is None else json.dumps(v)
                    lines.append(f"{pad}{k}: {sval}")
            return "\n".join(lines)
        elif isinstance(obj, list):
            lines = []
            for v in obj:
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.append(emit(v, indent + 1))
                else:
                    sval = "null" if v is None else json.dumps(v)
                    lines.append(f"{pad}- {sval}")
            return "\n".join(lines)
        else:
            return f"{pad}{json.dumps(obj)}"

    return emit(d) + "\n"


def proprio_stream(spec: ProprioSpec) -> Iterable[Dict[str, float]]:
    """Generate synthetic proprio rows with timestamps and joint telemetry."""
    period = 1.0 / max(spec.hz, 1e-6)
    n_steps = int(spec.duration_s * spec.hz)
    t0_ns = time.time_ns()
    # simple deterministic signals per joint
    for i in range(n_steps):
        t_ns = t0_ns + int(i * period * 1e9)
        row: Dict[str, float] = {"t_ns": t_ns, "frame_idx": i + 1}
        for j in range(spec.n_joints):
            # small bounded waveforms
            pos = 0.5 * (j + 1) * 0.01 * ((i % 50) / 50.0)
            vel = pos * 2.0
            row[f"joint_pos_{j}"] = float(pos)
            row[f"joint_vel_{j}"] = float(vel)
        row["gripper_width"] = 0.02 + 0.005 * ((i % 20) / 20.0)

        # End-effector wrench (fx, fy, fz, tx, ty, tz) synthetic signals
        fx = 5.0 * ((i % 30) / 30.0)
        fy = 4.0 * (((i + 10) % 35) / 35.0)
        fz = 6.0 * (((i + 20) % 40) / 40.0)
        tx = 0.2 * ((i % 25) / 25.0)
        ty = 0.25 * (((i + 5) % 28) / 28.0)
        tz = 0.3 * (((i + 12) % 32) / 32.0)
        row["ee_wrench_fx"] = float(fx)
        row["ee_wrench_fy"] = float(fy)
        row["ee_wrench_fz"] = float(fz)
        row["ee_wrench_tx"] = float(tx)
        row["ee_wrench_ty"] = float(ty)
        row["ee_wrench_tz"] = float(tz)

        # Actions: EE delta position (m), delta rotation (rad), gripper delta width (m)
        dx = 0.001 * ((i % 10) - 5) / 5.0
        dy = 0.001 * (((i + 3) % 10) - 5) / 5.0
        dz = 0.001 * (((i + 6) % 10) - 5) / 5.0
        droll = 0.005 * ((i % 8) - 4) / 4.0
        dpitch = 0.005 * (((i + 2) % 8) - 4) / 4.0
        dyaw = 0.005 * (((i + 4) % 8) - 4) / 4.0
        dgw = 0.0005 * ((i % 6) - 3) / 3.0
        # Store as floats in the row
        row["action_ee_dpos_x"] = float(dx)
        row["action_ee_dpos_y"] = float(dy)
        row["action_ee_dpos_z"] = float(dz)
        row["action_ee_drot_rx"] = float(droll)
        row["action_ee_drot_ry"] = float(dpitch)
        row["action_ee_drot_rz"] = float(dyaw)
        row["action_gripper_dwidth"] = float(dgw)
        yield row

        # sleep to approximate real-time without blocking tests too long
        time.sleep(min(0.002, period))


def _idx_str(i: int, width: int = 6) -> str:
    return f"{i:0{width}d}"


def generate_random_images(
    ep_dir: Path, rows: List[Dict[str, float]], cams: List[str]
) -> None:
    """Generate and save random RGB and depth frames for each camera and row."""
    rgb_h, rgb_w = 480, 848
    d_h, d_w = 240, 424
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    for r in rows:
        idx = int(r["frame_idx"])
        name = _idx_str(idx)
        for cam in cams:
            rgb_dir = ep_dir / "cams" / cam / "rgb"
            depth_dir = ep_dir / "cams" / cam / "depth"

            # RGB: uint8 [0,255], shape (H,W,3)
            rgb = np.random.randint(0, 256, size=(rgb_h, rgb_w, 3), dtype=np.uint8)
            cv2.imwrite(str(rgb_dir / f"{name}.png"), rgb, png_params)

            # Depth: uint16 mm, shape (H,W)
            depth = np.random.randint(0, 5000, size=(d_h, d_w), dtype=np.uint16)
            cv2.imwrite(str(depth_dir / f"{name}.png"), depth, png_params)


def save_parquet(rows: List[Dict[str, float]], out_path: Path) -> bool:
    """Save rows to Parquet using pyarrow if available. Returns success."""
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        # Ensure consistent column order
        if not rows:
            table = pa.table({})
        else:
            cols = sorted(rows[0].keys(), key=lambda k: (k != "t_ns", k))
            arrays = {k: [r.get(k) for r in rows] for k in cols}
            table = pa.table(arrays)
        pq.write_table(table, out_path)
        return True
    except Exception:
        return False


def save_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    """Fallback: save rows as CSV if Parquet isn't available."""
    if not rows:
        out_path.write_text("")
        return
    cols = sorted(rows[0].keys(), key=lambda k: (k != "t_ns", k))
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def append_index_jsonl(
    ep_dir: Path, rows: List[Dict[str, float]], cams: List[str]
) -> None:
    """Append entries to global index.jsonl with per-camera streams."""
    idx_path = ep_dir / "index.jsonl"
    with idx_path.open("a") as f:
        for r in rows:
            name = _idx_str(int(r["frame_idx"]))
            global_t = int(r["t_ns"])  # using proprio clock as global
            streams: Dict[str, Dict[str, object]] = {}
            for cam in cams:
                streams[cam] = {
                    "rgb": f"cams/{cam}/rgb/{name}.png",
                    "depth": f"cams/{cam}/depth/{name}.png",
                    "t_ns": global_t,
                    "dt_ns_offset": 0,
                }
            f.write(
                json.dumps(
                    {
                        "frame_idx": int(r["frame_idx"]),
                        "t_ns": global_t,
                        "streams": streams,
                    }
                )
            )
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create episode dir and record proprio to Parquet",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./dataset_root"),
        help="Dataset root directory.",
    )
    parser.add_argument("--hz", type=float, default=50.0, help="Sample rate")
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Recording duration in seconds",
    )
    parser.add_argument(
        "--joints",
        type=int,
        default=6,
        help="Number of joints in synthetic data",
    )
    args = parser.parse_args()

    spec = ProprioSpec(n_joints=args.joints, hz=args.hz, duration_s=args.duration)

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    # Define cameras (can be overridden later to match your rig)
    cams = ["overhead_azure", "wrist_d405"]
    ep_dir = init_episode(root, cams)

    rows = list(proprio_stream(spec))

    # Save states (Parquet preferred, CSV fallback)
    parquet_ok = save_parquet(rows, ep_dir / "state.parquet")
    if not parquet_ok:
        save_csv(rows, ep_dir / "state.csv")
        print(
            f"Parquet not available; wrote CSV fallback to {ep_dir / 'state.csv'}",
        )
    else:
        print(f"Wrote Parquet to {ep_dir / 'state.parquet'}")

    # Save random images per frame for each camera
    generate_random_images(ep_dir, rows, cams)

    append_index_jsonl(ep_dir, rows, cams)
    print(f"Episode created at: {ep_dir}")


if __name__ == "__main__":
    main()
