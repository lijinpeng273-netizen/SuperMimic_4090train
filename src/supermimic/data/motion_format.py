"""H1 motion/checkpoint identity and validation shared by all entry points."""

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

from supermimic.robot_dimensions import NJ, REF_DIM
from supermimic.robots.h1 import H1_XML, H1_JOINT_NAMES


@lru_cache(maxsize=1)
def asset_sha256() -> str:
    digest = hashlib.sha256(H1_XML.read_bytes())
    for path in sorted((H1_XML.parent / "meshes").rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(H1_XML.parent)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def robot_metadata() -> dict:
    return {"robot": "h1", "asset_sha256": asset_sha256(), "joint_names": list(H1_JOINT_NAMES)}


def checkpoint_identity() -> dict:
    from supermimic.robots.h1 import default_pos, stiffness, damping, effort_limit, H1_ACTION_SCALE

    return {**robot_metadata(), "default_pos": default_pos, "stiffness": stiffness,
            "damping": damping, "effort_limit": effort_limit, "action_scale": H1_ACTION_SCALE}


def validate_checkpoint(checkpoint: dict) -> None:
    if checkpoint.get("robot_identity") != checkpoint_identity():
        raise ValueError("Checkpoint is not for this H1 asset/control configuration. "
                         "Old G1 checkpoints cannot be resumed as H1; start a new H1 run.")


def validate_identity(data, source) -> None:
    for key, expected in robot_metadata().items():
        if key not in data or np.asarray(data[key]).tolist() != expected:
            raise ValueError(f"{source}: missing/mismatched H1 {key}; convert with the H1 converter first")


def validate_motion(data, source: str | Path = "motion") -> list[str]:
    """Validate a named, 50 Hz H1 clip; return its per-file body order."""
    validate_identity(data, source)
    fps = np.asarray(data["fps"]).reshape(-1)
    if fps.size != 1 or not np.isclose(fps[0], 50.0):
        raise ValueError(f"{source}: expected 50 Hz (environment control dt = 0.02 s)")
    names = [str(n) for n in data["body_names"]]
    if len(names) != len(set(names)) or "pelvis" not in names:
        raise ValueError(f"{source}: body_names must be unique and contain pelvis")
    n = len(data["joint_pos"])
    if n < 2:
        raise ValueError(f"{source}: a training clip needs at least two frames")
    for key, tail in (("joint_pos", (NJ,)), ("joint_vel", (NJ,)),
                      ("body_pos_w", (len(names), 3)), ("body_quat_w", (len(names), 4)),
                      ("body_lin_vel_w", (len(names), 3)), ("body_ang_vel_w", (len(names), 3))):
        arr = np.asarray(data[key])
        if arr.shape != (n, *tail) or not np.isfinite(arr).all():
            raise ValueError(f"{source}: {key} must be finite with shape {(n, *tail)}, got {arr.shape}")
    if not np.allclose(np.linalg.norm(data["body_quat_w"], axis=-1), 1, atol=2e-3):
        raise ValueError(f"{source}: body quaternions must be unit wxyz quaternions")
    return names


def validate_packed(pack: dict, source) -> None:
    import torch

    validate_identity(pack, source)
    ref = pack["ref"]
    if pack.get("fps") != 50 or pack.get("ref_dim") != REF_DIM or ref.ndim != 2 or ref.shape[1] != REF_DIM or not torch.isfinite(ref).all():
        raise ValueError(f"{source}: invalid H1 packed reference states or fps")
    names, starts, lengths = pack["names"], pack["starts"], pack["lengths"]
    if not names or len(set(names)) != len(names) or not len(names) == len(starts) == len(lengths):
        raise ValueError(f"{source}: invalid clip tables")
    end = 0
    for start, length in zip(starts, lengths, strict=True):
        if int(start) != end or int(length) < 2:
            raise ValueError(f"{source}: invalid clip offset/length")
        end += int(length)
    if end != len(ref) or not torch.allclose(ref[:, 3:7].float().norm(dim=-1), torch.ones(end), atol=2e-3):
        raise ValueError(f"{source}: invalid packed frame count or quaternion")
