"""SuperMimic tracking environment on mjlab (plan Phase 2.3).

Differences from mjlab's BeyondMimic re-implementation (``mjlab.tasks.tracking``):
  * observations: paper Table A.1 proprioception only (no anchor pose, no base lin vel),
    10-frame history (term-major), plus a separate ``goal`` group with K future frames;
  * action: ``NoisyJointPositionAction`` (clean action observed, noise only executed);
  * multi-clip command with return-based reset sampling and clip-end time-outs;
  * rewards: self-collision term removed, joint-limit weight -2;
  * terminations: pelvis height / tilt vs reference, no end-effector term, clip-end time-out;
  * events: no push; friction and encoder bias re-sampled per episode; COM at startup;
  * a ``ReplayRecorder`` streams transitions into a GPU replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# from mjlab.asset_zoo.robots import H1_ACTION_SCALE, get_h1_robot_cfg
from supermimic.robots.h1 import H1_ACTION_SCALE, get_h1_robot_cfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic.data.recorder import ReplayRecorderCfg
from supermimic.envs.actions import NoisyJointPositionActionCfg
from supermimic.envs.mdp import observations as sm_obs
from supermimic.envs.mdp import terminations as sm_term
from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg
from supermimic.envs.mdp.rewards import TrackingRewardCfg
from supermimic.envs.mdp.rewards import motion_arm_joint_position_error_exp, arm_action_target_limits, motion_planar_drift, motion_heading

class QuietSimulationCfg(SimulationCfg):
    """Legacy wrapper; keep overflow warnings enabled for H1 contact validation."""

    def apply_wp_opt(self, wp_opt) -> None:
        super().apply_wp_opt(wp_opt)
        # wp_opt.warn_overflow = 0
        wp_opt.warn_overflow = True  # 显示所有接触/约束容量溢出，避免隐藏 H1 配置问题。

# 修改成 H1 机器人的配置
TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link_ball_hand",
    "right_shoulder_roll_link",
    "right_elbow_link_ball_hand",
)
ANCHOR_BODY = "torso_link"
ROOT_BODY = "pelvis"
FOOT_GEOM_REGEX = r"^(left|right)_ankle_link_collision$"


@dataclass
class SuperMimicEnvCfgArgs:
    npz_files: list[str]
    num_envs: int = 4096
    future_frames: int = 10
    history_length: int = 10
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1   # 噪声标准差
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"  # return | uniform | start 三种采样方式
    replay_capacity: int = 1000
    record_transitions: bool = True
    seed: int | None = 0
    root_height_threshold: float = 0.25 # 根部高度相对参考值允许的最大误差
    root_tilt_threshold: float = 0.8    # 根部倾斜判据的阈值
    joint_limit_weight: float = -2.0    # 超出软关节限位的惩罚权重
    reward_cfg: TrackingRewardCfg | None = None
    bin_frames: int = 50    # 帧率
    compact_library: bool = False
    anchor_obs: bool = False  # add BeyondMimic anchor-relative pose (9) to the goal group (plan D5 alt.)
    debug_vis: bool = False  # draw the reference ghost in the viewer (scripts/play.py)
    extra_command_kwargs: dict = field(default_factory=dict)


def make_env_cfg(a: SuperMimicEnvCfgArgs) -> ManagerBasedRlEnvCfg:
    # ---------------------------------------------------------------- observations
    noise = (lambda lo, hi: Unoise(n_min=lo, n_max=hi)) if a.obs_noise else (lambda lo, hi: None)   # 定义噪声函数
    policy_terms = {    # 在观测中加上范围噪声
        "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, params={"biased": True}, noise=noise(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, noise=noise(-0.5, 0.5)),
        "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel, noise=noise(-0.2, 0.2)),
        "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity, noise=noise(-0.05, 0.05)),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    observations = {
        "policy": ObservationGroupCfg(  # 本体观测 10 * 63
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
            history_length=a.history_length,
            flatten_history_dim=True,
        ),
        "goal": ObservationGroupCfg(    # 目标参考观测
            terms={
                "future_frames": ObservationTermCfg(func=sm_obs.goal_future_frames, params={"command_name": "motion"}),
                **(
                    {
                        "anchor_pos": ObservationTermCfg(func=track_mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=noise(-0.25, 0.25)),
                        "anchor_ori": ObservationTermCfg(func=track_mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=noise(-0.05, 0.05)),
                    }
                    if a.anchor_obs
                    else {}
                ),
            },
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
        ),
    }

    # ---------------------------------------------------------------- actions
    actions = { # 策略动作转换为关节目标
        "joint_pos": NoisyJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=H1_ACTION_SCALE,
            use_default_offset=True,
            noise_std=a.action_noise_std,
        )
    }

    # ---------------------------------------------------------------- commands
    # 设置重置时候的扰动值
    pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)}
    vel_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}
    commands = {
        "motion": MultiClipMotionCommandCfg(    # 参考动作命令
            npz_files=[str(f) for f in a.npz_files],
            anchor_body_name=ANCHOR_BODY,
            root_body_name=ROOT_BODY,
            body_names=TRACKED_BODIES,
            entity_name="robot",
            future_frames=a.future_frames,
            bin_frames=a.bin_frames,
            pose_range=pose_range if a.reset_noise else {},
            velocity_range=vel_range if a.reset_noise else {},
            joint_position_range=(-0.1, 0.1) if a.reset_noise else (0.0, 0.0),
            sampling_mode=a.sampling_mode,  # type: ignore[arg-type] 采样方式
            compact_library=a.compact_library,
            debug_vis=a.debug_vis,
            **a.extra_command_kwargs,
        )
    }

    # ---------------------------------------------------------------- events (plan D8)
    events: dict[str, EventTermCfg] = {}    # 物理参数和传感器参数的随机化
    if a.domain_randomization:
        events["base_com"] = EventTermCfg(
            mode="startup", # 什么时候触发事件，startup表示在环境启动时触发
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add", # 表示在质心部分加上坐标偏移
                "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        )
        events["encoder_bias"] = EventTermCfg(  # 编码器偏置，随机化传感器的读数
            mode="reset",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
        )
        events["foot_friction"] = EventTermCfg( # 摩擦系数随机化
            mode="reset",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        )
    else:
        # Evaluation protocol: foot friction fixed at 1.0, no other randomization.
        events["foot_friction"] = EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
                "operation": "abs",
                "ranges": (1.0, 1.0),
                "shared_random": True,
            },
        )

    # ---------------------------------------------------------------- rewards (Table A.2)
    rewards = { # 运动跟踪奖励函数 计算函数，权重，计算参数
        "motion_global_root_pos": RewardTermCfg(func=track_mdp.motion_global_anchor_position_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.3}),
        "motion_global_root_ori": RewardTermCfg(func=track_mdp.motion_global_anchor_orientation_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.4}),
        "motion_body_pos": RewardTermCfg(func=track_mdp.motion_relative_body_position_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.3}),
        "motion_body_ori": RewardTermCfg(func=track_mdp.motion_relative_body_orientation_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.4}),
        "motion_body_lin_vel": RewardTermCfg(func=track_mdp.motion_global_body_linear_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 1.0}),
        "motion_body_ang_vel": RewardTermCfg(func=track_mdp.motion_global_body_angular_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 3.14}),
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.1),
        "joint_limit": RewardTermCfg(func=base_mdp.joint_pos_limits, weight=a.joint_limit_weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
    }

    if a.reward_cfg is not None:
        r = a.reward_cfg
        if r.dt != 0.02:
            raise ValueError("H1 reward dt must match the 50 Hz control period")
        if r.planar_drift_scale <= 0 or r.w_planar_drift > 0 or r.w_heading < 0:
            raise ValueError("Global drift needs a positive scale, nonpositive cost weight and nonnegative heading weight")
        for name, weight, std in (
            ("motion_global_root_pos", r.w_root_pos, r.std_root_pos),
            ("motion_global_root_ori", r.w_root_ori, r.std_root_ori),
            ("motion_body_pos", r.w_body_pos, r.std_body_pos),
            ("motion_body_ori", r.w_body_ori, r.std_body_ori),
            ("motion_body_lin_vel", r.w_body_lin_vel, r.std_body_lin_vel),
            ("motion_body_ang_vel", r.w_body_ang_vel, r.std_body_ang_vel),
            ("action_rate_l2", r.w_action_rate, None),
            ("joint_limit", r.w_joint_limit, None),
        ):
            rewards[name].weight = weight
            if std is not None:
                rewards[name].params["std"] = std
        if r.w_arm_joint_pos:
            rewards["motion_arm_joint_pos"] = RewardTermCfg(
                func=motion_arm_joint_position_error_exp, weight=r.w_arm_joint_pos,
                params={"command_name": "motion", "std": r.std_arm_joint_pos},
            )
        if r.w_arm_target_limit:
            rewards["arm_target_limit"] = RewardTermCfg(
                func=arm_action_target_limits, weight=r.w_arm_target_limit,
                params={"command_name": "motion"},
            )
        if r.w_planar_drift:
            rewards["motion_global_planar_drift"] = RewardTermCfg(
                func=motion_planar_drift, weight=r.w_planar_drift,
                params={"command_name": "motion", "scale": r.planar_drift_scale},
            )
        if r.w_heading:
            rewards["motion_global_heading"] = RewardTermCfg(
                func=motion_heading, weight=r.w_heading, params={"command_name": "motion"},
            )

    # ---------------------------------------------------------------- terminations
    terminations = {    # 终止条件判断
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "clip_end": TerminationTermCfg(func=sm_term.clip_end, params={"command_name": "motion"}, time_out=True),
        "root_height": TerminationTermCfg(func=sm_term.bad_root_height, params={"command_name": "motion", "threshold": a.root_height_threshold}),
        "root_tilt": TerminationTermCfg(func=sm_term.bad_root_tilt, params={"command_name": "motion", "threshold": a.root_tilt_threshold}),
    }

    recorders = {}  # 配置交互数据记录器
    if a.record_transitions:
        recorders["replay"] = ReplayRecorderCfg(params={"capacity": a.replay_capacity, "foot_geom_regex": FOOT_GEOM_REGEX, "com_body_name": "torso_link"})

    cfg = ManagerBasedRlEnvCfg( # 指定使用 H1 机器人模型，设置场景、观测、动作和仿真参数。
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=a.num_envs, entities={"robot": get_h1_robot_cfg()}),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        recorders=recorders,
        # 设置观测视角
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="torso_link", distance=2.8, fovy=55.0, elevation=-5.0, azimuth=120.0),
        # 设置模拟参数，包括时间步长、迭代次数和最大接触数
        # H1 contacts reached the 10-iteration solver cap at 4096 environments.
        # Allow convergence without changing the timestep or solver tolerance.
        sim=QuietSimulationCfg(nconmax=35, njmax=250, mujoco=MujocoCfg(timestep=0.005, iterations=50, ls_iterations=50)),
        decimation=4,
        episode_length_s=a.episode_length_s,
        seed=a.seed,
    )
    return cfg

# 创建环境实例
def make_env(a: SuperMimicEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    return ManagerBasedRlEnv(make_env_cfg(a), device=device)
