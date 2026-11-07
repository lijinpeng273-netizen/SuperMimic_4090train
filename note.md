

# 训练命令 ：
cd /home/rl-4090/RL_Workspace/ljp_ws/SuperMimic_H1-main

./scripts/train_h1.sh \
  --run-name h1_gpu256 \
  --env.num-envs 4096 \
  --wm-trainer.batch-size 4096 \
  --policy-trainer.batch-size 4096 \
  --schedule.max-hours 10


# 推理策略命令：

cd /home/rl-4090/RL_Workspace/ljp_ws/SuperMimic_H1-main

./scripts/local.sh python scripts/play.py \
  --ckpt logs/h1_arm_fix_stage_011000/ckpt_latest.pt \
  --npz-dir data/npz/h1 \
  --clips dance1_subject3_last \
  --num-envs 1 \
  --device cuda:0 \
  --viewer native




查看这次 4 小时训练的最终模型，以 walk1_subject1 动作为例：

cd /home/rl-4090/RL_Workspace/ljp_ws/SuperMimic_H1-main

./scripts/local.sh python scripts/play.py \
  --ckpt logs/h1_4h_all/ckpt_latest.pt \
  --npz-dir data/npz/h1 \
  --clips walk1_subject1 \
  --num-envs 1 \
  --device cuda:0 \
  --viewer native

窗口中，实体机器人是策略控制的结果，半透明重影是参考动作。CPU 绑核和四线程限制会自动生效。

以后主要修改以下参数：

| 需求 | 修改参数 | 示例 |
|---|---|---|
| 查看新训练得到的策略 | --ckpt：模型文件路径 | logs/新训练名称/ckpt_latest.pt |
| 查看某个训练阶段 | --ckpt：选择编号检查点 | 本次的 ckpt_0032000.pt 表示第 32000 轮 |
| 播放其他参考动作 | --clips：NPZ 文件名，不带 .npz | dance1_subject3_last |
| 动作文件放在其他目录 | --npz-dir：NPZ 所在文件夹 | /你的路径/npz/h1 |

例如，播放 data/npz/h1/dance1_subject3_last.npz，只需将命令中的这一行改为：

  --clips dance1_subject3_last \
