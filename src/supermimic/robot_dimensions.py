"""H1 的统一维度定义（迁移指南第 3 章）。

假定浮动根部和 19 个单自由度受控关节，不含 not_use_joint。
训练环境、数据和网络共同使用本模块，避免不同路径的机器人尺寸不一致。
"""

NJ = 19 # 受控关节数（不含 not_use_joint）。
NQ = 7 + NJ  # 根位置 3 + 四元数 4 + 关节位置。
NV = 6 + NJ  # 根线速度 3 + 根角速度 3 + 关节速度。
STATE_DIM = NQ + NV
FEAT_DIM = 13 + 2 * NJ  # 高度 1 + 旋转表示 6 + 根速度 6 + 关节位置/速度。
DELTA_DIM = 12 + 2 * NJ  # 根位置/旋转/线速度/角速度增量各 3 维。
ACT_DIM = NJ    # 关节位置动作。
XI_DIM = 1 + NJ + 3  # 共享脚摩擦 + 关节编码器偏置 + 躯干质心偏移。
OBS_DIM = 3 * NJ + 6  # 关节位置/速度/上次动作 + 根角速度/重力方向。
GOAL_FEAT_DIM = 2 * NJ + 10  # 参考关节位置/速度 + 重力/角速度/线速度/高度。
REF_DIM = STATE_DIM # # 参考状态维度（与 STATE_DIM 相同）。

# 顺序：根位置、四元数、关节位置、根线速度、根角速度、关节速度。
STATE_SIZES = (3, 4, NJ, 3, 3, NJ)
# 顺序：dv_h、dw_b、dqd、dp_h、dth_b、dq。
DELTA_SIZES = (3, 3, NJ, 3, 3, NJ)
OBS_TERMS = (
    ("joint_pos", NJ),
    ("joint_vel", NJ),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("actions", NJ),
)
