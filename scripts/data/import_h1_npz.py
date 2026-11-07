"""Import an existing H1 NPZ, explicitly remapping its source joint order.

Source root positions must be metres in a Z-up world; root quaternions are wxyz.
Use the matching source policy's ONNX joint_names metadata OR a JSON list of names.
The caller must identify the pelvis index in the source NPZ (not its tracked-body list).
All velocities and body FK are recomputed at 50 Hz using this repository's H1 asset.
"""

from pathlib import Path
import json

import numpy as np
import tyro

from supermimic.data.motion_convert import resample_motion, save_motion_npz
from supermimic.robots.h1 import H1_JOINT_NAMES


def main(source: Path, out: Path, root_index: int, onnx: Path | None = None,
         joint_names_json: Path | None = None, root_height_offset: float = 0.0,
         overwrite: bool = False) -> None:
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; use --overwrite to replace it")
    if (onnx is None) == (joint_names_json is None):
        raise ValueError("Provide exactly one of --onnx and --joint-names-json")
    if onnx is not None:
        import onnx as onnx_lib

        model = onnx_lib.load(str(onnx), load_external_data=False)
        metadata = {p.key: p.value for p in model.metadata_props}
        value = metadata["joint_names"]
        names = json.loads(value) if value.strip().startswith("[") else value.split(",")
    else:
        names = json.loads(joint_names_json.read_text())
    names = [n.strip() for n in names]
    if len(names) != len(H1_JOINT_NAMES) or set(names) != set(H1_JOINT_NAMES):
        raise ValueError("Source joint_names must contain each of the 19 H1 joints exactly once")
    order = [names.index(n) for n in H1_JOINT_NAMES]
    with np.load(source, allow_pickle=False) as data:
        joint_pos = data["joint_pos"]
        pos, quat = data["body_pos_w"], data["body_quat_w"]
        if joint_pos.ndim != 2 or joint_pos.shape[1] != 19:
            raise ValueError("Source must be H1 joint_pos (T,19); G1 motions require retargeting")
        if pos.ndim != 3 or not 0 <= root_index < pos.shape[1] or quat.shape != (*pos.shape[:2], 4):
            raise ValueError("Invalid root_index or body pose arrays")
        fps_arr = np.asarray(data["fps"]).reshape(-1)
        if fps_arr.size != 1:
            raise ValueError("Source fps must be a scalar or one-element array")
        fps = float(fps_arr[0])
        raw = np.concatenate([pos[:, root_index], quat[:, root_index, [1, 2, 3, 0]], joint_pos[:, order]], axis=-1)
    # One constant translation preserves jumps; never snap every frame to the floor.
    raw[:, 2] += root_height_offset
    motion = resample_motion(raw, fps, 50.0)
    provenance = f"{source.resolve()}; joint_names={onnx or joint_names_json}; pelvis_index={root_index}; root_height_offset={root_height_offset}"
    save_motion_npz(out, motion, source=provenance, input_fps=fps)
    print(f"Imported {source.name}: {motion.num_frames} frames, 19 H1 joints, 50 Hz -> {out}")


if __name__ == "__main__":
    tyro.cli(main)
