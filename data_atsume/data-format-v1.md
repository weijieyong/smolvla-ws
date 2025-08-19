# Dataset layout format

## 1) Top-level structure

```
dataset_root/
  episode_2025-08-18T17-20-33Z/
    index.jsonl               # global timeline across all streams
    state.parquet             # proprio, actions, etc.
    cams/
      overhead_azure/
        rgb/                  # 000001.png, 000002.png, ...
        depth/                # 000001.png (uint16 mm) or 000001.exr (float m)
      wrist_d405/
        rgb/
        depth/
    calibration.yaml          # all intrinsics/extrinsics & depth scales here
    metadata.json             # episode-level info incl. task & prompt
```

* **One folder per episode/trajectory.** Human-inspectable and POSIX-friendly.
* **Single global index.** Stateless readers stream frames; converters can pack into HDF5/Zarr/WebDataset later.

## 2) File formats (by signal)

### A) RGB images

* **Preferred:** `PNG` (lossless; simplest).
* **Space-lean alternative:** `JPEG` (quality \~92–95).
* **Naming:** zero-padded frame indices (`000001.png`); timestamps live in `index.jsonl`.

### B) Depth images

* **Simplest & portable:** **16-bit PNG** (`uint16`) in **millimeters** (or decimillimeters).

  * Document unit/scale in `calibration.yaml` (e.g., `depth_unit: "mm"`, `depth_scale: 0.001` to convert to meters).
* **High-fidelity option:** **OpenEXR** (`.exr`) with **float32 meters**.
* **Avoid:** 8-bit depth (insufficient precision).

### C) Proprioception (joints, gripper width, F/T, etc.)

* **Preferred:** **Apache Parquet** (`state.parquet`)
  Columns (example):

  * `t_ns` (int64, UNIX ns)
  * `frame_idx` (int)
  * `joint_pos[j]`, `joint_vel[j]` (float32)
  * `gripper_width` (float32)
  * `ee_wrench_*` (float32)
  * `action_ee_dpos` (float32)
  * `action_ee_drot` (float32)
  * `action_gripper_dwidth` (float32)

## 3) Indexing & metadata

### A) `index.jsonl` (one line per global tick)

Each line maps the **global frame** to per-camera frames and times:

```json
{
  "frame_idx": 1234,
  "t_ns": 1723972830156790123,
  "streams": {
    "overhead_azure": {
      "rgb": "cams/overhead_azure/rgb/000742.png",
      "depth": "cams/overhead_azure/depth/000742.png",
      "t_ns": 1723972830156700000,
      "dt_ns_offset": -90123
    },
    "wrist_d405": {
      "rgb": "cams/wrist_d405/rgb/000811.png",
      "depth": "cams/wrist_d405/depth/000811.png",
      "t_ns": 1723972830156890000,
      "dt_ns_offset": 999877
    }
  }
}
```

* `t_ns`: global tick timestamp (int64 ns).
* For each camera: chosen frame paths, that frame’s native `t_ns`, and `dt_ns_offset = cam_t − global_t`.
* If a camera has no match within tolerance, set that stream entry to `null`.

### B) `calibration.yaml` (single file at episode root)

Holds all camera intrinsics/extrinsics, distortion, and depth scale:

```yaml
version: 1
frames:
  base_link: {}
  ee_link: {}

cameras:
  overhead_azure:
    model: pinhole
    resolution: [1280, 720]
    intrinsics: {fx: ..., fy: ..., cx: ..., cy: ...}
    distortion: {model: radtan, params: [k1, k2, p1, p2, k3]}
    depth_unit: "mm"          # if depth saved as uint16 PNG
    depth_scale: 0.001        # mm -> m
    parent_frame: base_link
    T_parent_camera:
      translation_m: [x, y, z]
      quaternion_xyzw: [qx, qy, qz, qw]

  wrist_d405:
    model: pinhole
    resolution: [640, 480]
    intrinsics: {...}
    distortion: {...}
    depth_unit: "mm"
    depth_scale: 0.001
    parent_frame: ee_link
    T_parent_camera: {...}
```

* **Static rigs:** `parent_frame: base_link`.
* **Wrist cams:** `parent_frame: ee_link`; per-tick pose can be derived via FK from `state.parquet`.

### C) `metadata.json` (minimal single-task episode)

Include task info and sync strategy:

```json
{
  "episode_id": "2025-08-18T17-20-33Z",
  "units": {"length": "m", "angle": "rad", "time": "ns"},
  "task_name": "insert_type_g_plug",
  "nl_prompt": "Insert the three-prong Type G plug fully into the socket until flush.",
  "primary_clock": "proprio",
  "sync_strategy": "nearest",
  "max_skew_ns": 50000000,
  "cameras": ["overhead_azure", "wrist_d405", "front_left"]
}
```

---

# Why this works (and converts cleanly later)

* **LeRobot:** Iterate `index.jsonl`, load PNGs/EXRs, join with `state.parquet`, emit `obs.image`, `obs.depth`, `obs.proprio`, `action`, `timestamp`.
* **LIBERO:** Read the same trio and pack into HDF5 (episode-per-file or consolidated).
* **Other containers:** Folder+JSONL maps cleanly to WebDataset, TFRecords, Zarr, or HDF5.

---

# Practical conventions (keep it simple, future-proof)

* **Timestamps:** use **int64 nanoseconds** everywhere.
* **Units/frames:** meters, radians, seconds; declare once in `metadata.json`.
* **Depth scale:** always documented in `calibration.yaml`.
* **Integrity (optional):** per-frame `sha256` in `index.jsonl`.
* **Growth path:** later shard episodes (WebDataset) or pack to HDF5/Zarr with no schema change.

---

# Minimal choices (shortest path today)

* **RGB:** PNG
* **Depth:** 16-bit PNG (mm)
* **Proprio:** Parquet
* **Index:** JSONL
* **Calibration:** single `calibration.yaml` at episode root
