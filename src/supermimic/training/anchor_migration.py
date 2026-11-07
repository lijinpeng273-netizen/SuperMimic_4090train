"""Lossless input expansion when resuming a legacy policy with anchor feedback."""
from __future__ import annotations

import torch

from supermimic.robot_dimensions import FEAT_DIM, GOAL_FEAT_DIM


def _insert_columns(value, offset):
    return torch.cat((value[:, :offset], value.new_zeros(value.shape[0], 9), value[:, offset:]), dim=1)


def migrate_anchor_inputs(checkpoint, policy, critic, anchor_obs: bool) -> bool:
    """Pad only the two affected layers and their Adam moments; retain all learning.

    New inputs initially have zero influence. In the critic they are inserted
    between the old goal and xi/root-error features, not at the end of the input.
    Other architecture changes are deliberately left to strict load_state_dict.
    """
    old_env = checkpoint["config"]["env"]
    old_anchor = old_env.get("anchor_obs", False)
    if old_anchor == anchor_obs:
        return False
    old_dim = checkpoint["config"]["policy"]["goal_dim"]
    if old_anchor or not anchor_obs or old_dim != old_env["future_frames"] * GOAL_FEAT_DIM or policy.cfg.goal_dim != old_dim + 9:
        raise ValueError("Only legacy goal -> same goal plus 9 anchor inputs can be migrated automatically")
    for name, model, weight_key, scale_key, offset in (
        ("policy", policy, "encoders.motion.mlp.net.0.weight", "encoders.motion.scale", old_dim),
        ("critic", critic, "mlp.net.0.weight", "goal_scale", FEAT_DIM + old_dim),
    ):
        state = checkpoint[name]
        old_shape = state[weight_key].shape
        expanded = _insert_columns(state[weight_key], offset)
        if expanded.shape != model.state_dict()[weight_key].shape:
            raise ValueError(f"Unsupported {name} architecture change during anchor migration")
        state[weight_key] = expanded
        state[scale_key] = torch.cat((state[scale_key], state[scale_key].new_ones(9)))
        parameter_names = [k for k, _ in model.named_parameters()]
        opt = checkpoint[name + "_opt"]
        ids = [i for group in opt["param_groups"] for i in group["params"]]
        if len(ids) != len(parameter_names):
            raise ValueError(f"Unexpected {name} optimizer layout")
        moments = opt["state"].get(ids[parameter_names.index(weight_key)], {})
        for key, value in moments.items():
            if isinstance(value, torch.Tensor) and value.shape == old_shape:
                moments[key] = _insert_columns(value, offset)
    return True
