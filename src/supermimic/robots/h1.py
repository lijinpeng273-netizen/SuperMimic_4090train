"""19 关节 H1 资产和位置控制配置（迁移指南第 4 章）。

CPU 模型加载不依赖 mjlab；只有创建训练实体配置时才导入 mjlab。
PD 参数和动作缩放用于初始仿真联调，尚未验证 H1 跟踪训练的稳定性。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco

from supermimic.robot_dimensions import NJ, NQ, NV

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg

# 源码仓库布局下的路径；打包部署时需要一并分发 XML 和 meshes。
H1_XML = Path(__file__).resolve().parents[3] / "h1_data/h1_training.xml"

# 与整理后的 XML 关节树顺序一致，不沿用原 motor 通道排列。
# 关节顺序：左腿 5、右腿 5、腰部 1、左臂 4、右臂 4，共 19 个关节。
H1_JOINT_NAMES = (
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

# q_target = q_default + scale * action；此值不是动作范围限制。
H1_ACTION_SCALE = {name: 0.25 for name in H1_JOINT_NAMES}
ARM_JOINT_INDICES = tuple(i for i, name in enumerate(H1_JOINT_NAMES) if "shoulder" in name or "elbow" in name)


# 以下列表均按 H1_JOINT_NAMES 排列：左腿 5、右腿 5、腰部 1、左臂 4、右臂 4。
# 原 XML home 关键帧的 19 个实际关节角，单位 rad。
default_pos: list[float] = [
    *[0.0, 0.0, -0.4, 0.8, -0.4],  # 左腿
    *[0.0, 0.0, -0.4, 0.8, -0.4],  # 右腿
    *[0.0],  # 腰部
    *[0.0, 0.0, 0.0, 0.0],  # 左臂
    *[0.0, 0.0, 0.0, 0.0],  # 右臂
]

# 位置控制刚度 Kp，单位 N m/rad。
stiffness: list[float] = [
    *[150.0, 150.0, 150.0, 200.0, 40.0],  # 左腿
    *[150.0, 150.0, 150.0, 200.0, 40.0],  # 右腿
    *[300.0],  # 腰部
    *[100.0, 100.0, 100.0, 50.0],  # 左臂
    *[100.0, 100.0, 100.0, 50.0],  # 右臂
]

# 位置控制阻尼 Kd，单位 N m s/rad。
damping: list[float] = [
    *[2.0, 2.0, 2.0, 4.0, 2.0],  # 左腿
    *[2.0, 2.0, 2.0, 4.0, 2.0],  # 右腿
    *[3.0],  # 腰部
    *[2.0, 2.0, 2.0, 2.0],  # 左臂
    *[2.0, 2.0, 2.0, 2.0],  # 右臂
]

# 力矩上限，单位 N m；沿用 h1_data/h1.xml 中保留的原 motor 配置。
effort_limit: list[float] = [
    *[200.0, 200.0, 200.0, 300.0, 40.0],  # 左腿
    *[200.0, 200.0, 200.0, 300.0, 40.0],  # 右腿
    *[200.0],  # 腰部
    *[40.0, 40.0, 18.0, 18.0],  # 左臂
    *[40.0, 40.0, 18.0, 18.0],  # 右臂
]

# 加载已编译模型
def load_h1_model(xml_path: str | Path | None = None) -> mujoco.MjModel:
    """加载 CPU 模型，用于数据转换/FK，并检查关节类型和顺序。"""
    model = mujoco.MjModel.from_xml_path(str(xml_path or H1_XML))   # xml文件的路径，下面三个if语句是对模型的检查
    if model.njnt != NJ + 1 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("H1 应包含一个根部 free joint 和 19 个实际关节。")
    names = tuple(model.joint(i).name for i in range(1, model.njnt))
    if names != H1_JOINT_NAMES:
        raise ValueError(f"H1 关节顺序不符：{names}")
    if (model.nq, model.nv) != (NQ, NV):
        raise ValueError(f"H1 模型尺寸不符：nq={model.nq}, nv={model.nv}")
    if any(model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE for i in range(1, model.njnt)):
        raise ValueError("H1 的非根部关节必须全部为 hinge joint。")
    return model


def get_h1_spec() -> mujoco.MjSpec:
    """每次返回新的可编辑模型；执行器由 mjlab 构建实体时添加。"""
    return mujoco.MjSpec.from_file(str(H1_XML))


def foot_sole_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Lowest world Z of the two H1 foot collision meshes after mj_kinematics.

    Unlike G1's capsules, H1 uses meshes. MuJoCo's compiled mesh vertices and
    geom_xmat/geom_xpos include the mesh's principal-axis transformation.
    """
    import numpy as np

    heights = []
    for side in ("left", "right"):
        geom = model.geom(f"{side}_ankle_link_collision").id
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Expected an H1 foot collision mesh")
        mesh = model.geom_dataid[geom]
        first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[first:first + count]
        rotation = data.geom_xmat[geom].reshape(3, 3)
        heights.append(np.min(vertices @ rotation[2] + data.geom_xpos[geom, 2]))
    return float(min(heights))

# 定义机器人资产来源，初始状态和控制参数
def get_h1_robot_cfg() -> EntityCfg:
    """构造独立的 H1 配置，避免多个环境配置之间共享可变实例。"""
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    actuators = []
    for name, kp, kd, effort in zip(    # 创建一个空列表，收集每一个关节的控制器配置
        H1_JOINT_NAMES, stiffness, damping, effort_limit, strict=True
    ):
        actuators.append(
            BuiltinPositionActuatorCfg(
                target_names_expr=(name,),
                stiffness=kp,
                damping=kd,
                effort_limit=effort,
                # 显式沿用 XML 的关节惯量和被动阻尼；被动阻尼不等于 PD Kd。
                armature=0.1,
                viscous_damping=1.0,
            )
        )

    return EntityCfg(
        spec_fn=get_h1_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.98),
            joint_pos=dict(zip(H1_JOINT_NAMES, default_pos, strict=True)),
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(actuators),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
