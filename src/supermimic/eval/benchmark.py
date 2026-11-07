"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires before the clip ends. Normalised return = sum of per-step
rewards / (max step reward * number of steps).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.data.motion_format import validate_checkpoint
from supermimic.robots.h1 import ARM_JOINT_INDICES, H1_JOINT_NAMES
from mjlab.utils.lab_api.math import quat_error_magnitude
from supermimic.kinematics.rotations import heading_yaw, wrap_to_pi


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)
    tracking: dict | None = None  # unweighted errors; comparable when training rewards change
    evaluation_complete: bool = True  # false only if the evaluation budget ran out
    stop_reason: str = "clip_end"


def arm_tracking_errors(command) -> dict[str, torch.Tensor]:
    """Measure each arm, since whole-body averages can hide a stuck shoulder."""
    joint_error = (command.robot_joint_pos - command.joint_pos).abs()
    result = {f"joint/{H1_JOINT_NAMES[i]}/mae_rad": joint_error[:, i] for i in ARM_JOINT_INDICES}
    for side in ("left", "right"):
        ids = [i for i in ARM_JOINT_INDICES if H1_JOINT_NAMES[i].startswith(side)]
        result[f"{side}_arm_joint_mae_rad"] = joint_error[:, ids].mean(-1)
        body = command.cfg.body_names.index(f"{side}_elbow_link_ball_hand")
        result[f"{side}_elbow_position_mae_m"] = torch.linalg.vector_norm(
            command.robot_body_pos_w[:, body] - command.body_pos_relative_w[:, body], dim=-1)
    return result


def tracking_errors(command, action) -> dict[str, torch.Tensor]:
    """Fixed, unweighted measurements of all 19 joints and 12 tracked bodies."""
    error = (command.robot_joint_pos - command.joint_pos).abs()
    velocity_error = (command.robot_joint_vel - command.joint_vel).abs()
    result = arm_tracking_errors(command)
    for i, name in enumerate(H1_JOINT_NAMES):
        prefix = f"joint/{name}"
        result[f"{prefix}/mae_rad"] = error[:, i]
        result[f"{prefix}/rmse_rad"] = error[:, i].square()  # square root after averaging
        result[f"{prefix}/max_abs_rad"] = error[:, i]
        result[f"{prefix}/velocity_mae_rad_s"] = velocity_error[:, i]
        result[f"{prefix}/within_10deg_fraction"] = (error[:, i] <= math.radians(10)).float()
    result["all_joint_mae_rad"] = error.mean(-1)
    result["all_joint_rmse_rad"] = error.square().mean(-1)
    for side, ids in (("left", range(5)), ("right", range(5, 10))):
        result[f"{side}_leg_joint_mae_rad"] = error[:, list(ids)].mean(-1)
    body_pos = torch.linalg.vector_norm(command.body_pos_relative_w - command.robot_body_pos_w, dim=-1)
    body_ori = quat_error_magnitude(command.body_quat_relative_w, command.robot_body_quat_w)
    for i, name in enumerate(command.cfg.body_names):
        result[f"body/{name}/position_mae_m"] = body_pos[:, i]
        result[f"body/{name}/orientation_mae_rad"] = body_ori[:, i]
    result["body_position_mae_m"] = body_pos.mean(-1)
    result["root_position_mae_m"] = torch.linalg.vector_norm(command.root_pos_w - command.robot_root_pos_w, dim=-1)
    result["root_orientation_mae_rad"] = quat_error_magnitude(command.root_quat_w, command.robot_root_quat_w)
    result["root_planar_position_mae_m"] = torch.linalg.vector_norm(
        command.root_pos_w[:, :2] - command.robot_root_pos_w[:, :2], dim=-1)
    result["root_heading_mae_rad"] = wrap_to_pi(
        heading_yaw(command.root_quat_w) - heading_yaw(command.robot_root_quat_w)).abs()
    # World-frame velocity distinguishes sideways/backwards drift from yaw alone.
    ri = command.cfg.body_names.index(command.cfg.root_body_name)
    ref_vel = command.body_lin_vel_w[:, ri, :2]
    robot_vel = command.robot_body_lin_vel_w[:, ri, :2]
    result["root_planar_velocity_mae_m_s"] = torch.linalg.vector_norm(robot_vel - ref_vel, dim=-1)
    ref_speed, robot_speed = ref_vel.norm(dim=-1), robot_vel.norm(dim=-1)
    result["root_planar_speed_mae_m_s"] = (robot_speed - ref_speed).abs()
    moving = (ref_speed > .1) & (robot_speed > .1)
    angle = wrap_to_pi(torch.atan2(robot_vel[:, 1], robot_vel[:, 0]) - torch.atan2(ref_vel[:, 1], ref_vel[:, 0])).abs()
    result["root_motion_direction_mae_rad"] = angle * moving
    result["root_motion_direction_valid_fraction"] = moving.float()
    term = command._env.action_manager.get_term("joint_pos")
    target = action * term._scale + term._offset
    limits = command.robot.data.joint_pos_limits
    outside = (target < limits[..., 0]) | (target > limits[..., 1])
    result["action/target_outside_hard_limits_fraction"] = outside.float().mean(-1)
    result["action/arm_target_outside_hard_limits_fraction"] = outside[:, ARM_JOINT_INDICES].float().mean(-1)
    result["action/mean_abs"] = action.abs().mean(-1)
    return result


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint(ck)
    pc = ck["config"]["policy"]
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, ck["config"]


@torch.no_grad()
def run_benchmark(policy: Policy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int = 10, verbose: bool = True, anchor_obs: bool = False, deadline_unix: float | None = None) -> list[ClipResult]:
    if not npz_files:
        raise ValueError("Benchmark needs at least one H1 clip")
    num_envs = min(num_envs or 1024, len(npz_files))
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        num_envs=num_envs,
        future_frames=future_frames,
        history_length=policy.cfg.history,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
        max_r = 0.1  # paper reward upper bound per step (dt * 5)

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        tracking_names = list(tracking_errors(cmd, torch.zeros_like(cmd.joint_pos)))
        tracking_sum = torch.zeros(num_envs, len(tracking_names), device=device)
        tracking_max = torch.zeros_like(tracking_sum)
        peak = torch.tensor([k.endswith("/max_abs_rad") for k in tracking_names], device=device)
        results: dict[int, ClipResult] = {}
        remaining = n_clips
        started = last_progress = time.time()
        loop_step = 0

        def finish(e, ci, failed=False, complete=True):
            st = int(steps[e])
            measurements = {}
            for i, key in enumerate(tracking_names):
                value = float(tracking_max[e, i] if key.endswith("/max_abs_rad") else tracking_sum[e, i] / max(st, 1))
                if key.endswith("rmse_rad"):
                    value = math.sqrt(max(value, 0.0))
                measurements[key] = value
            valid = measurements["root_motion_direction_valid_fraction"]
            measurements["root_motion_direction_mae_rad"] /= max(valid, 1e-12)
            return ClipResult(name=lib.clips[ci].name, frames=lib.clips[ci].length,
                steps=st, norm_return=float(ret_sum[e]) / (max_r * max(st, 1)),
                failed=failed, fail_step=st if failed else -1,
                terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                tracking=measurements if st else None, evaluation_complete=complete,
                stop_reason="budget" if not complete else ("termination" if failed else "clip_end"))

        while remaining > 0:
            if loop_step % 50 == 0 and deadline_unix is not None and time.time() >= deadline_unix:
                break
            # Sample before stepping: auto-reset otherwise replaces the terminal pose.
            act = policy(obs["policy"], obs["goal"])
            errors = tracking_errors(cmd, act)
            values = torch.stack([errors[k] for k in tracking_names], dim=-1)
            tracking_sum += values * active[:, None]
            tracking_max = torch.maximum(tracking_max, values * peak * active[:, None])
            obs, rew, term, tout, _ = env.step(act)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            steps += 1
            done = (term | tout) & active
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        results[ci] = finish(e, ci, bool(term[e]))
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                tracking_sum[ids] = 0.0
                tracking_max[ids] = 0.0
                steps[ids] = 0
            loop_step += 1
            if verbose and time.time() - last_progress >= 30:
                print(f"  evaluated {len(results)}/{n_clips}, simulation step {loop_step}, {time.time()-started:.0f}s elapsed", flush=True)
                last_progress = time.time()
        if remaining:
            for e in active.nonzero().flatten().tolist():
                ci = int(current[e])
                if ci < n_clips and ci not in results:
                    results[ci] = finish(e, ci, complete=False)
            for ci, clip in enumerate(lib.clips):
                if ci not in results:
                    results[ci] = ClipResult(clip.name, clip.length, 0, 0.0, False, -1,
                        evaluation_complete=False, stop_reason="budget")
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Length-weighted return (paper Fig. 3a uses the benchmark aggregate; weighting by steps).
    tot_steps = sum(r.steps for r in results)
    w_ret = sum(r.norm_return * r.steps for r in results) / max(tot_steps, 1)
    completed = sum(r.evaluation_complete and not r.failed and r.steps >= r.frames - 1 for r in results)
    out = {"clips": n, "failed": failed, "mean_norm_return": mean_ret, "step_weighted_norm_return": w_ret,
        "completed_clips": completed, "completion_rate": completed / max(n, 1),
        "frame_coverage": sum(min(r.steps, r.frames) for r in results) / max(sum(r.frames for r in results), 1),
        "evaluation_complete": all(r.evaluation_complete for r in results)}
    if results and all(r.terms is not None for r in results):
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    if results and all(r.tracking is not None for r in results):
        out["tracking"] = {}
        for k in results[0].tracking:
            if k.endswith("/max_abs_rad"):
                value = max(r.tracking[k] for r in results)
            elif k == "root_motion_direction_mae_rad":
                count = sum(r.tracking["root_motion_direction_valid_fraction"] * r.steps for r in results)
                value = sum(r.tracking[k] * r.tracking["root_motion_direction_valid_fraction"] * r.steps for r in results) / max(count, 1e-12)
            elif k.endswith("rmse_rad"):
                value = math.sqrt(sum(r.tracking[k]**2 * r.steps for r in results) / max(tot_steps, 1))
            else:
                value = sum(r.tracking[k] * r.steps for r in results) / max(tot_steps, 1)
            out["tracking"][k] = value
    return out


def save_results(results: list[ClipResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"protocol": {"reward": "original fixed 8 terms; max per step 0.1",
        "noise_and_domain_randomization": False, "sampling": "clip start, once per clip",
        "tracking_samples": "pre-step states until termination or clip end; failed clips cover only their executed prefix",
        "heading": "absolute wrapped world yaw error (not full quaternion angle)",
        "motion_direction": "mean world XY velocity angle only where BOTH speeds exceed 0.1 m/s; zero if no valid frames, read valid_fraction",
        "joint_names": list(H1_JOINT_NAMES)},
        "summary": summarize(results), "clips": [asdict(r) for r in results]}, indent=1))
