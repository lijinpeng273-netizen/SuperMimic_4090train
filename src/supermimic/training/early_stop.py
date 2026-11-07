"""Conservative quality gate on completed fixed-reward evaluations."""
import math


DEFAULT_POLICY = {
    "enabled": True,
    "evaluation_interval": 5000,
    "consecutive_evaluations": 2,
    "min_mean_return": 0.85,
    "min_clip_return": 0.80,
    "max_arm_joint_mae_rad": 0.15,
    "max_elbow_position_mae_m": 0.08,
    "expected_clips": ["dance1_subject3_last", "default_pose_150frames"],
}


def quality_goal(history, policy):
    """Require distinct scheduled evaluations and full completion of every clip."""
    if not policy.get("enabled", False):
        return False, "提前停止未启用"
    count = int(policy["consecutive_evaluations"])
    if count < 2:
        raise ValueError("At least two consecutive evaluations are required")
    if len(history) < count:
        return False, "达标评估次数尚不足"
    interval = int(policy.get("evaluation_interval", 5000))
    if interval <= 0:
        raise ValueError("Evaluation interval must be positive")
    recent = history[-count:]
    iterations = [r["iteration"] for r in recent]
    if any(n <= 0 or n % interval for n in iterations) or any(b - a != interval for a, b in zip(iterations, iterations[1:])):
        return False, f"必须是连续、不同轮数的正式 {interval} 轮评估"
    for record in recent:
        evaluation = record["evaluation"]
        clips, summary = evaluation["clips"], evaluation["summary"]
        if sorted(c["name"] for c in clips) != sorted(policy["expected_clips"]):
            return False, "评估动作集不完整"
        if summary["failed"] != 0 or summary["clips"] != len(clips):
            return False, "仍存在跟踪失败"
        scores = [float(c["norm_return"]) for c in clips]
        means = [float(summary["mean_norm_return"]), float(summary["step_weighted_norm_return"])]
        if not all(math.isfinite(v) for v in scores + means):
            return False, "评估包含非有限数值"
        if min(means) < policy["min_mean_return"] or min(scores) < policy["min_clip_return"]:
            return False, "归一化回报尚未达标"
        if any(c["failed"] or c["frames"] < 2 or c["steps"] < c["frames"] - 1 for c in clips):
            return False, "必须完整完成每段动作"
        for clip in clips:
            tracking = clip.get("tracking") or {}
            joints = [v for k, v in tracking.items() if k.startswith("joint/") and k.endswith("/mae_rad") and ("shoulder" in k or "elbow" in k)]
            positions = [tracking.get(f"{side}_elbow_position_mae_m", float("inf")) for side in ("left", "right")]
            if len(joints) != 8 or not all(math.isfinite(v) for v in joints + positions):
                return False, "缺少有效的左右臂跟踪评估，不能仅凭全身回报提前停止"
            if max(joints) > policy.get("max_arm_joint_mae_rad", 0.15) or max(positions) > policy.get("max_elbow_position_mae_m", 0.08):
                return False, f"{clip['name']} 的手臂跟踪误差尚未达标"
    return True, f"连续 {count} 次正式评估全部动作完整完成，回报和左右臂跟踪误差均达标；轮数 {iterations}"
