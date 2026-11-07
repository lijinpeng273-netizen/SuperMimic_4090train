"""Read / write the full simulator state (qpos | qvel) as one 51-d tensor."""

from __future__ import annotations

import torch

from supermimic.robot_dimensions import STATE_DIM, NQ, NV


class StateIO:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        # States are stored relative to each env's origin so that they are directly comparable
        # with the (origin-free) motion library and translation-invariant across envs.
        self.origins = env.scene.env_origins.clone()  # (N, 3)
        assert self.sim.data.qpos.shape[-1] == NQ and self.sim.data.qvel.shape[-1] == NV, (
            self.sim.data.qpos.shape,
            self.sim.data.qvel.shape,
        )

    def read(self) -> torch.Tensor:
        """(N, 51) copy of [qpos | qvel]."""
        # Index with [:] to get plain tensors out of mjlab's TorchArray proxies (torch.cat on the
        # proxies recurses inside their __torch_function__).
        s = torch.cat([self.sim.data.qpos[:], self.sim.data.qvel[:]], dim=-1).clone()
        s[:, 0:3] -= self.origins
        return s

    @torch.no_grad()
    def write(self, env_ids: torch.Tensor, state: torch.Tensor, forward: bool = True) -> None:
        """Overwrite qpos/qvel of ``env_ids`` and refresh derived quantities."""
        state = state.to(self.sim.data.qpos.dtype).clone()
        state[:, 0:3] += self.origins[env_ids]
        self.sim.data.qpos[env_ids] = state[:, :NQ]
        self.sim.data.qvel[env_ids] = state[:, NQ:]
        if hasattr(self.sim.data, "qacc_warmstart"):
            self.sim.data.qacc_warmstart[env_ids] = 0.0
        if forward:
            self.sim.forward()

"""
H1 状态布局
| 切片     | 内容           | 坐标约定                 |
| ------- | ------------- | ------------------------ |
| `0:3`   | 根位置         | 相对环境原点的世界轴方向 |
| `3:7`   | 根四元数       | wxyz                     |
| `7:26`  | 19 个关节位置  | H1 统一关节顺序          |
| `26:19` | 根线速度       | 世界坐标系               |
| `19:32` | 根角速度       | 根部坐标系               |
| `32:51` | 19 个关节速度  | H1 统一关节顺序          |
"""
