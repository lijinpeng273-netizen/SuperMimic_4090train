"""Pack filtered npz clips into one compact library file (fp16 51-d reference states).

  uv run python scripts/data/build_library.py --npz-dir data/npz/h1 --manifest data/npz/h1/manifest_filtered.csv --out data/library/h1_compact.pt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.data.motion_format import robot_metadata, validate_motion, validate_packed
from supermimic.robot_dimensions import REF_DIM


def main(npz_dir: Path, out: Path, manifest: Path | None = None, workers: int = 16) -> None:
    files = discover_npz(npz_dir, manifest=manifest)
    if not files:
        raise FileNotFoundError(f"No H1 clips in {npz_dir}")
    print(f"{len(files)} clips")
    from concurrent.futures import ThreadPoolExecutor

    def load(f):
        with np.load(f) as d:
            names = validate_motion(d, f)
            r = names.index("pelvis")
            ref = np.concatenate([d["body_pos_w"][:, r], d["body_quat_w"][:, r], d["joint_pos"], d["body_lin_vel_w"][:, r], d["body_ang_vel_w"][:, r], d["joint_vel"]], axis=1)
        return f.stem, ref.astype(np.float16)

    with ThreadPoolExecutor(workers) as ex:
        items = list(tqdm(ex.map(load, files), total=len(files)))
    names = [n for n, _ in items]
    lengths = [int(r.shape[0]) for _, r in items]
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    ref = torch.from_numpy(np.concatenate([r for _, r in items], 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    pack = {**robot_metadata(), "fps": 50, "ref_dim": REF_DIM, "ref": ref, "names": names, "starts": starts, "lengths": lengths}
    validate_packed(pack, out)
    torch.save(pack, out)
    print(f"saved {out}: {ref.shape[0]} frames ({ref.shape[0]/50/3600:.2f} h), {ref.numel()*2/1e9:.2f} GB fp16")


if __name__ == "__main__":
    tyro.cli(main)
