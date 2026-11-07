"""Conservative comparisons and readable reports for staged H1 training."""
from copy import deepcopy
import math

from supermimic.robots.h1 import H1_JOINT_NAMES


GLOBAL_LIMITS = (("root_position_mae_m", .25), ("root_orientation_mae_rad", .10),
                 ("root_planar_position_mae_m", .25), ("root_heading_mae_rad", .10),
                 ("root_planar_velocity_mae_m_s", .15))


def regressions(current, baseline):
    """Compare fixed evaluations; increased coverage can expose harder motion frames."""
    if not current["summary"]["evaluation_complete"]:
        return ["评估未完整执行，不能接受候选"]
    if {c["name"] for c in current["clips"]} != {c["name"] for c in baseline["clips"]}:
        return ["动作集合不一致"]
    a, b = current["summary"], baseline["summary"]
    reasons = []
    if a["frame_coverage"] < b["frame_coverage"] - .05 or a["frame_coverage"] < b["frame_coverage"] * .75 - .002:
        reasons.append("总体动作覆盖率明显下降（超过 5 个百分点或约 25%）")
    if a["completed_clips"] < b["completed_clips"] - 1:
        reasons.append("完整完成的动作减少至少 2 个")
    comparable = a["frame_coverage"] <= b["frame_coverage"] * 1.1 + .002
    if comparable:
        if a["step_weighted_norm_return"] < b["step_weighted_norm_return"] - .04:
            reasons.append("固定奖励回报下降超过 0.04")
        for key, allowance in (("all_joint_mae_rad", .02), ("body_position_mae_m", .02)):
            if a["tracking"][key] > b["tracking"][key] * 1.25 + allowance:
                reasons.append(f"{key} 明显恶化（超过原值 25% 加 {allowance}）")
        for key, allowance in GLOBAL_LIMITS:
            if key in a["tracking"] and key in b["tracking"] and a["tracking"][key] > b["tracking"][key] * 1.25 + allowance:
                reasons.append(f"全局跟踪退步：{key} 超过原值 25% 加 {allowance}")
    old = {c["name"]: c for c in baseline["clips"]}
    joint_regressions = []
    for clip in current["clips"]:
        prev = old[clip["name"]]
        # Longer successful prefixes are not directly comparable with short failures.
        if not prev["failed"] and prev["steps"] >= prev["frames"] - 1 and clip["failed"] and clip["steps"] < .75 * prev["steps"]:
            reasons.append(f"{clip['name']} 原先完整完成，现在提前丢失超过四分之一动作")
        if clip["steps"] > prev["steps"] * 1.1 + 2:
            continue
        if clip["steps"] >= prev["steps"] * .9 - 2:
            for key, allowance in GLOBAL_LIMITS:
                before, after = prev["tracking"].get(key), clip["tracking"].get(key)
                if before is not None and after is not None and after > before * 1.5 + 2 * allowance:
                    reasons.append(f"{clip['name']} 全局退步：{key} {before:.3f}→{after:.3f}")
        for joint in H1_JOINT_NAMES:
            key = f"joint/{joint}/mae_rad"
            before, after = prev["tracking"][key], clip["tracking"][key]
            if after > before * 1.5 + .15:
                joint_regressions.append(f"{clip['name']}:{joint} {before:.3f}→{after:.3f} rad")
    if joint_regressions:
        reasons.append("局部关节明显退步：" + "；".join(joint_regressions[:8]))
    return reasons


def improving(current, baseline):
    if regressions(current, baseline):
        return False
    a, b = current["summary"], baseline["summary"]
    return (a["step_weighted_norm_return"] > b["step_weighted_norm_return"] + .01
        or a["frame_coverage"] > b["frame_coverage"] + .02
        or a["frame_coverage"] > b["frame_coverage"] * 1.2 + .002
        or a["completed_clips"] > b["completed_clips"]
        or a["tracking"]["all_joint_mae_rad"] < b["tracking"]["all_joint_mae_rad"] * .95 - .005
        or any(key in a["tracking"] and key in b["tracking"] and a["tracking"][key] < b["tracking"][key] * .9 - allowance
               for key, allowance in GLOBAL_LIMITS))


def propose(settings, history):
    """Change a single parameter family only after a comparable plateau."""
    candidate = deepcopy(settings)
    if len(history) < 2:
        return candidate, "首个完整评估建立基线，保持参数。"
    current, previous = history[-1], history[-2]
    if improving(current["evaluation"], previous["evaluation"]):
        return candidate, "完成率、固定回报或关节跟踪仍在改善，保持参数。"
    error = current["metrics"].get("wm/online_nse_8step")
    if error is None:
        return candidate, "缺少有效的世界模型多步误差，保持参数。"
    wm, pi, reward = candidate["wm_trainer"], candidate["policy_trainer"], candidate["rewards"]
    if error > .1 and wm["batches_per_update"] < 20:
        wm["batches_per_update"] += 5
        return candidate, "表现停滞且 8 步预测误差较高，世界模型每次更新增加 5 批。"
    if error > .15 and pi["horizon"] > 16:
        pi["horizon"] -= 8
        return candidate, "预测仍不可靠，策略想象长度缩短 8 步，最短 16。"
    if error > .05 and wm["window"] < 16:
        wm["window"] += 4
        return candidate, "多步误差仍偏高，世界模型监督窗口增加 4 步，最长 16。"
    if error < .05 and pi["horizon"] < 32:
        pi["horizon"] += 8
        return candidate, "模型误差降低，逐步恢复 32 步想象长度。"
    s = current["evaluation"]["summary"]
    t = s["tracking"]
    if error < .05:
        if (t.get("root_heading_mae_rad", 0) > math.radians(20) or t.get("root_planar_position_mae_m", 0) > .5):
            if not candidate["env"].get("anchor_obs", False):
                candidate["env"]["anchor_obs"] = True
                return candidate, "全局轨迹/航向误差偏大，启用相对位置朝向反馈；兼容迁移旧模型输入。"
            if reward.get("w_heading", 0) < 1.0:
                reward["w_heading"] = min(1.0, max(.5, reward.get("w_heading", 0) * 1.2))
                return candidate, "全局轨迹/航向停滞，增加宽范围航向奖励；按固定全局指标验证，退步回退。"
        if t["action/arm_target_outside_hard_limits_fraction"] > .05 and abs(reward["w_arm_target_limit"]) < 1:
            reward["w_arm_target_limit"] = max(-1., reward["w_arm_target_limit"] * 1.2)
            return candidate, "手臂目标越界比例超过 5%，目标限位惩罚增大 20%。"
        if max(t["left_arm_joint_mae_rad"], t["right_arm_joint_mae_rad"]) > .15 and reward["w_arm_joint_pos"] < 2:
            reward["w_arm_joint_pos"] = min(2., reward["w_arm_joint_pos"] * 1.2)
            return candidate, "手臂平均角度误差偏高，手臂跟踪奖励增加 20%。"
        if t["body_position_mae_m"] > .15 and reward["w_body_pos"] < 2:
            reward["w_body_pos"] = min(2., reward["w_body_pos"] * 1.2)
            return candidate, "全身位置跟踪停滞，身体位置奖励增加 20%。"
        if s.get("terms", {}).get("action_rate_l2", 0) < -.01 and abs(reward["w_action_rate"]) < .2:
            reward["w_action_rate"] = max(-.2, reward["w_action_rate"] * 1.2)
            return candidate, "动作变化惩罚偏大，平滑惩罚幅度增加 20%。"
    return candidate, "未达到明确调整条件，保持参数。"


def parameter_changes(before, after):
    rows = ["| 参数 | 调整前 | 调整后 |", "|---|---:|---:|"]
    for section, values in after.items():
        for key, value in values.items():
            old = before.get(section, {}).get(key)
            if old != value:
                rows.append(f"| `{section}.{key}` | {old} | {value} |")
    return "\n".join(rows) if len(rows) > 2 else "参数保持不变。"


def evaluation_markdown(result):
    s = result["summary"]
    lines = [f"动作数：{s['clips']}；完整完成：{s['completed_clips']}；跟踪终止：{s['failed']}；"
        f"帧覆盖率：{s['frame_coverage']:.1%}；固定奖励回报：{s['step_weighted_norm_return']:.4f}。",
        "", "评估是否完整执行：" + str(s["evaluation_complete"]), "",
        "失败动作的误差仅覆盖已经执行的部分，必须结合完成率阅读。", "",
        "| 动作 | 执行步数/帧数 | 完成 | 固定回报 | 19关节 MAE(°) | RMSE(°) | 右臂 MAE(°) | 躯干漂移(cm) |",
        "|---|---:|---|---:|---:|---:|---:|---:|"]
    for c in result["clips"]:
        t = c.get("tracking")
        if not t:
            lines.append(f"| {c['name']} | {c['steps']}/{c['frames']} | 未完成评估 | — | — | — | — | — |")
            continue
        done = c["evaluation_complete"] and not c["failed"] and c["steps"] >= c["frames"] - 1
        lines.append(f"| {c['name']} | {c['steps']}/{c['frames']} | {'是' if done else c['stop_reason']} | "
            f"{c['norm_return']:.4f} | {math.degrees(t['all_joint_mae_rad']):.2f} | "
            f"{math.degrees(t['all_joint_rmse_rad']):.2f} | {math.degrees(t['right_arm_joint_mae_rad']):.2f} | "
            f"{100*t['root_position_mae_m']:.2f} |")
    lines += ["", "全局误差不做平移或航向对齐；速度方向仅统计双方速度均 >0.1 m/s 的帧。", "",
              "| 动作 | XY误差(m) | 航向误差(°) | XY速度误差(m/s) | 速度方向误差(°) | 有效速度帧比例 |",
              "|---|---:|---:|---:|---:|---:|"]
    for c in result["clips"]:
        t = c.get("tracking") or {}
        if "root_heading_mae_rad" in t:
            direction = f"{math.degrees(t['root_motion_direction_mae_rad']):.2f}" if t['root_motion_direction_valid_fraction'] else "无有效帧"
            lines.append(f"| {c['name']} | {t['root_planar_position_mae_m']:.3f} | {math.degrees(t['root_heading_mae_rad']):.2f} | "
                         f"{t['root_planar_velocity_mae_m_s']:.3f} | {direction} | {t['root_motion_direction_valid_fraction']:.1%} |")
    for c in result["clips"]:
        t = c.get("tracking")
        if not t:
            continue
        lines += ["", f"<details><summary>{c['name']}：全部 19 个关节和身体部位</summary>", "",
            "| 关节 | MAE(°) | RMSE(°) | 最大误差(°) | 速度 MAE(rad/s) | 误差≤10°比例 |",
            "|---|---:|---:|---:|---:|---:|"]
        for joint in H1_JOINT_NAMES:
            p = f"joint/{joint}"
            lines.append(f"| {joint} | {math.degrees(t[p+'/mae_rad']):.2f} | {math.degrees(t[p+'/rmse_rad']):.2f} | "
                f"{math.degrees(t[p+'/max_abs_rad']):.2f} | {t[p+'/velocity_mae_rad_s']:.3f} | {t[p+'/within_10deg_fraction']:.1%} |")
        lines += ["", "| 身体部位 | 位置误差(cm) | 朝向误差(°) |", "|---|---:|---:|"]
        for key in t:
            if key.startswith("body/") and key.endswith("/position_mae_m"):
                body = key.split("/")[1]
                lines.append(f"| {body} | {100*t[key]:.2f} | {math.degrees(t[f'body/{body}/orientation_mae_rad']):.2f} |")
        lines += ["", f"目标角越过硬限位的关节-帧比例：{t['action/target_outside_hard_limits_fraction']:.2%}；"
            f"手臂：{t['action/arm_target_outside_hard_limits_fraction']:.2%}。", "", "</details>"]
    return "\n".join(lines)
