"""Run bounded H1 stages, evaluate all motions/joints, and conservatively resume."""
from __future__ import annotations

import argparse
from copy import deepcopy
import dataclasses
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time

from supermimic.training.supervision import (
    propose, regressions, improving, parameter_changes, evaluation_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
INTERVAL = 5000
TRIAL_STEPS = 250


def stamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def configure(cfg, settings):
    for section, values in settings.items():
        target = getattr(cfg, section)
        for key, value in values.items():
            if key not in {f.name for f in dataclasses.fields(target)}:
                raise ValueError(f"Unknown setting {section}.{key}")
            if value is not None and (isinstance(getattr(target, key), Path) or key in {"npz_dir", "manifest", "library"}):
                value = Path(value)
            setattr(target, key, value)
    return cfg


def worker(args):
    if not Path(args.session).resolve().is_relative_to(ROOT):
        raise ValueError("Training outputs must stay within the H1 project")
    import torch
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    settings = json.loads(Path(args.settings).read_text())
    cfg = configure(TrainConfig(run_name=args.run_name, log_root=Path(args.session)), settings)
    cfg.resume = Path(args.resume) if args.resume else None
    cfg.schedule.total_iterations = 320_000
    cfg.schedule.stop_at_iteration = args.boundary
    cfg.schedule.deadline_unix = args.deadline
    cfg.schedule.checkpoint_interval = 5000
    cfg.schedule.log_interval = 50
    cfg.schedule.max_hours = None
    trainer = Trainer(cfg)
    try:
        # Changing a reward must affect physical and imagined rewards together.
        assert trainer.wm.heads.reward_cfg == cfg.rewards
        trainer.run()
        for model in (trainer.wm, trainer.policy, trainer.critic):
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError("Nonfinite model parameters; stop supervision")
        write_json(trainer.log_dir / "completed.json", {
            "iteration": trainer.iteration, "updates": trainer.num_updates,
            "finished": stamp(), "finite_parameters": True,
        })
    finally:
        trainer.env.close()
        if trainer.writer is not None:
            trainer.writer.close()


def metrics(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(Path(run) / "tb")).Reload()
    result = {}
    for tag in acc.Tags()["scalars"]:
        values = acc.Scalars(tag)
        if values:
            value = float(values[-1].value)
            if not math.isfinite(value):
                raise RuntimeError(f"Nonfinite training metric: {tag}")
            result[tag] = value
    return result


def supervise(args):
    session = Path(args.session).resolve()
    if not session.is_relative_to(ROOT):
        raise ValueError("Supervision outputs must stay within the H1 project")
    (ROOT / "logs").mkdir(exist_ok=True)
    lock = (ROOT / "logs/h1_supervision.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    session.mkdir(parents=True, exist_ok=args.continue_session)
    report = ROOT / "docs" / f"{session.name}.md"
    from supermimic.envs.mdp.rewards import TrackingRewardCfg
    from supermimic.envs.mdp.motion_library import discover_npz
    files = discover_npz(ROOT / "data/npz/h1")
    if not files:
        raise ValueError("No H1 NPZ motions found")
    settings = {
        "data": {"npz_dir": "data/npz/h1", "clips": [p.stem for p in files],
                 "limit": None, "manifest": None, "library": None},
        "env": {"num_envs": 4096, "replay_capacity": 256, "anchor_obs": True},
        "wm_trainer": {"batch_size": 4096, "batches_per_update": 10, "window": 8, "compile": "default"},
        "policy_trainer": {"batch_size": 4096, "horizon": 32, "compile": "default", "lr": 2e-4},
        "rewards": dataclasses.asdict(TrackingRewardCfg(w_arm_joint_pos=1.0, w_arm_target_limit=-0.2,
            w_planar_drift=-0.5, w_heading=0.5)),
    }
    active_settings = session / "parameters.json"
    started = time.time()
    deadline = started + args.hours * 3600
    state = {"pid": os.getpid(), "started": stamp(), "deadline_unix": deadline,
        "deadline_local": dt.datetime.fromtimestamp(deadline).astimezone().isoformat(timespec="seconds"),
        "status": "starting", "iteration": 0, "report": str(report), "checkpoint": None,
        "evaluation_interval": INTERVAL, "quality_early_stop_enabled": False,
        "motion_count": len(files), "hours": args.hours}
    history, baseline, best = [], None, None
    if args.continue_session:
        state = json.loads((session / "state.json").read_text())
        if state["status"] not in ("stopped", "failed"):
            raise RuntimeError("Only interrupted or failed sessions can be continued")
        deadline = state["deadline_unix"]
        settings = json.loads(active_settings.read_text())
        history = json.loads((session / "history.json").read_text()) if (session / "history.json").exists() else []
        baseline = json.loads((session / "validated.json").read_text()) if (session / "validated.json").exists() else None
        best = json.loads((session / "best_validated.json").read_text()) if (session / "best_validated.json").exists() else None
        state.update(pid=os.getpid(), status="starting", quality_early_stop_enabled=False)
        for key in ("finished", "child_pid", "error"):
            state.pop(key, None)
    write_json(active_settings, settings)
    # The old quality watcher cannot stop this session; this supervisor has no quality-stop branch.
    write_json(session / "early_stop_policy.json", {"enabled": False, "evaluation_interval": INTERVAL,
        "reason": "User requested the full three-hour budget, with no quality-based early stop"})
    clip_list = session / "clips.txt"
    clip_list.write_text("\n".join(settings["data"]["clips"]) + "\n")
    active = None
    last_eval_seconds = 300.0

    def note(text):
        with report.open("a") as handle:
            handle.write(f"\n{text}\n")
        print(text if len(text) <= 6000 else text[:1400] + f"\n[完整逐动作/逐关节记录已写入 {report}]", flush=True)

    if not args.continue_session:
        report.write_text(f"# H1 全动作集 {args.hours:g} 小时训练监管记录\n\n开始：{state['started']}\n\n"
            f"截止：{state['deadline_local']}\n\n数据：data/npz/h1，共 {len(files)} 个动作；本次从头训练。\n\n"
            "总时长包含编译、训练、评估和候选验证。每到累计 5000 轮，训练进程保存并退出，"
            "评估完成后用检查点和下一阶段参数重新启动。效果好也不提前停止。\n\n"
            "运行方式：终端显示自动监督器执行的实际命令和日志；监督器按以下明确规则执行，"
            "不是聊天模型持续在线判断。\n\n"
            "评估：全部动作，全部 19 个关节的角度 MAE/RMSE/最大误差、速度误差、10°以内比例；"
            "12 个身体部位的位置和朝向误差、躯干漂移、动作完成率、覆盖率、目标角越界比例。"
            "固定使用原始八项奖励、关闭噪声和随机化，每个动作从首帧播放。"
            "跟踪失败后的未执行帧不伪装为已评估，必须结合覆盖率判断误差。\n\n"
            "调整：改善时保持参数；停滞时每次最多调整一类参数，候选先续训 250 轮再完整评估。"
            "覆盖率下降超过 5 个百分点或约 25%、完成动作减少至少 2 个、相近覆盖下固定回报下降超过 0.04、"
            "全身误差明显增大或某个动作的某个关节明显恶化，都触发回退。"
            "覆盖更长的动作可能暴露更难的帧，比较时会考虑这一点。"
            "严重退步优先回退已验证模型并降低策略学习率，不因一次波动直接丢弃全部训练。"
            "这些是保守启发式规则，不保证收敛。\n\n"
            "配置：CPU 16–23，四项数值库线程数均为 4；RTX 4090；4096 个环境，"
            "世界模型/策略 batch size 均 4096，回放容量 256；保留网络、控制周期、原终止条件，"
            "启用已验证的手臂跟踪奖励和目标角越界惩罚。\n\n"
            "parameters.json 保存下一阶段的实际参数；每次评估和调整追加于本文件，"
            "所有检查点、每个动作的完整指标和控制台日志保存在同名 logs 目录。\n")
        note("初始参数：\n```json\n" + json.dumps(settings, ensure_ascii=False, indent=2) + "\n```")
    else:
        note(f"## 接续 — {stamp()}\n\n保留原截止时间 {state['deadline_local']}；质量提前停止保持关闭。")

    def run(command, log, timeout):
        nonlocal active
        note("执行命令：\n```bash\n" + shlex.join(command) + "\n```")
        with Path(log).open("w") as sink:
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
            active = child
            state["child_pid"] = child.pid
            write_json(session / "state.json", state)
            def pump():
                for line in child.stdout:
                    print(line, end="", flush=True)
                    sink.write(line)
                    sink.flush()
            thread = threading.Thread(target=pump, daemon=True)
            thread.start()
            try:
                code = child.wait(timeout=max(1, timeout))
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                raise
            finally:
                thread.join(timeout=10)
                active = None
                state.pop("child_pid", None)
            if code:
                raise RuntimeError(f"Command exited {code}: {shlex.join(command)}")

    def unique_label(prefix):
        label, retry = prefix, 0
        while (session / label).exists() or (session / f"{label}.parameters.json").exists():
            retry += 1
            label = f"{prefix}_retry{retry:02d}"
        return label

    def train(label, target, resume, config):
        path = session / f"{label}.parameters.json"
        write_json(path, config)
        cmd = ["./scripts/local.sh", "python", "-B", "-u", "scripts/supervise_h1.py", "worker",
            "--settings", str(path), "--session", str(session), "--run-name", label,
            "--boundary", str(target), "--deadline", str(deadline)]
        if resume:
            cmd += ["--resume", str(resume)]
        state.update(status="training", stage=label, target_iteration=target)
        run(cmd, session / f"{label}.console.log", deadline - time.time() + 90)
        folder = session / label
        completed = json.loads((folder / "completed.json").read_text())
        return str(folder / "ckpt_latest.pt"), completed["iteration"], metrics(folder)

    def evaluate(checkpoint, label):
        nonlocal last_eval_seconds
        out = session / f"{label}.evaluation.json"
        state["status"] = "evaluating"
        t0 = time.time()
        run(["./scripts/local.sh", "python", "-B", "-u", "scripts/eval_benchmark.py",
             "--ckpt", str(checkpoint), "--npz-dir", "data/npz/h1", "--clip-list", str(clip_list),
             "--num-envs", str(len(settings["data"]["clips"])), "--out", str(out),
             "--deadline-unix", str(deadline)], session / f"{label}.evaluation.log", deadline - time.time() + 60)
        last_eval_seconds = time.time() - t0
        result = json.loads(out.read_text())
        expected = set(settings["data"]["clips"])
        if {c["name"] for c in result["clips"]} != expected:
            raise RuntimeError("Evaluation did not include the complete motion set")
        for clip in result["clips"]:
            values = [clip["norm_return"], *(clip.get("tracking") or {}).values()]
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError("Nonfinite evaluation metric")
        return result

    def remember(record, config, checkpoint):
        nonlocal baseline, best
        baseline = {"checkpoint": checkpoint, "iteration": record["iteration"],
                    "settings": deepcopy(config), "record": record}
        write_json(session / "validated.json", baseline)
        state["last_validated_checkpoint"] = checkpoint
        if best is None or improving(record["evaluation"], best["record"]["evaluation"]):
            best = deepcopy(baseline)
            write_json(session / "best_validated.json", best)
            state["best_checkpoint"] = checkpoint
            state["best_score"] = record["evaluation"]["summary"]["step_weighted_norm_return"]

    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        checkpoint, iteration = state.get("checkpoint"), state["iteration"]
        while time.time() < deadline and iteration < 320000:
            target = min(320000, (iteration // INTERVAL + 1) * INTERVAL)
            label = unique_label(f"stage_{target:06d}")
            checkpoint, iteration, data = train(label, target, checkpoint, settings)
            state.update(iteration=iteration, checkpoint=checkpoint)
            write_json(session / "state.json", state)
            if iteration != target or deadline - time.time() < 5:
                note(f"## 到达时间预算 — {stamp()}\n\n保存第 {iteration} 轮模型：{checkpoint}。"
                     "本阶段未到下一个正式评估点，最后模型未重新完成全动作评估；已验证的模型另外保留。")
                break
            result = evaluate(checkpoint, label)
            record = {"kind": "stage", "iteration": iteration, "time": stamp(),
                      "evaluation": result, "metrics": data, "checkpoint": checkpoint}
            history.append(record)
            note(f"## 第 {iteration} 轮评估 — {stamp()}\n\n" + evaluation_markdown(result) +
                 "\n\n训练指标：\n```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```")
            if not result["summary"]["evaluation_complete"]:
                record["decision"] = "budget_incomplete_evaluation"
                write_json(session / "history.json", history)
                note("时间预算内未完成全部评估，禁止据此调参，保留最后训练模型和已验证模型。")
                break
            bad = regressions(result, baseline["record"]["evaluation"]) if baseline else []
            if bad:
                record["decision"] = "rollback"
                old_settings = deepcopy(settings)
                settings = deepcopy(baseline["settings"])
                settings["policy_trainer"]["lr"] = max(1e-5, settings["policy_trainer"]["lr"] * .5)
                checkpoint, iteration = baseline["checkpoint"], baseline["iteration"]
                note("明显退步，回退到上一个完整验证的检查点；仅降低策略学习率后继续。\n\n" +
                     "\n".join("- " + r for r in bad) + "\n\n" + parameter_changes(old_settings, settings))
            else:
                record["decision"] = "accepted"
                remember(record, settings, checkpoint)
                accepted_history = [r for r in history if r.get("kind") == "stage" and r.get("decision") == "accepted"]
                candidate, reason = propose(settings, accepted_history)
                note("调整判断：" + reason + "\n\n" + parameter_changes(settings, candidate))
                needed = max(600., 2 * last_eval_seconds + 180.)
                if candidate != settings and deadline - time.time() > needed:
                    original, original_it, original_settings = checkpoint, iteration, deepcopy(settings)
                    trial_label = unique_label(f"trial_{iteration:06d}")
                    try:
                        trial, trial_it, trial_data = train(trial_label, iteration + TRIAL_STEPS, checkpoint, candidate)
                        if trial_it != iteration + TRIAL_STEPS or deadline - time.time() < 5:
                            note("候选训练未完整完成验证预算，保持原参数与已验证检查点。")
                        else:
                            trial_eval = evaluate(trial, trial_label)
                            bad_trial = regressions(trial_eval, result)
                            tr = {"kind": "trial", "iteration": trial_it, "time": stamp(),
                                  "evaluation": trial_eval, "metrics": trial_data, "checkpoint": trial,
                                  "decision": "rollback" if bad_trial else "accepted"}
                            history.append(tr)
                            note(f"### 候选验证 — {trial_it} 轮\n\n" + evaluation_markdown(trial_eval))
                            if not bad_trial:
                                settings, checkpoint, iteration = candidate, trial, trial_it
                                remember(tr, settings, checkpoint)
                                note("候选通过多指标检查，使用新参数及候选检查点继续。")
                            else:
                                note("候选回退：" + "；".join(bad_trial))
                    except Exception as exc:
                        checkpoint, iteration, settings = original, original_it, original_settings
                        note(f"候选验证失败，恢复原参数和权重：{exc}")
                elif candidate != settings:
                    note("剩余时间不足以完成候选训练和全动作验证，本次保持参数。")
            write_json(session / "history.json", history)
            write_json(active_settings, settings)
            state.update(iteration=iteration, checkpoint=checkpoint)
            write_json(session / "state.json", state)
        state["status"] = "finished"
        state["stop_reason"] = "time_budget" if time.time() >= deadline - 5 else "iteration_limit_or_evaluation_budget"
        note(f"## 结束 — {stamp()}\n\n停止原因：{state['stop_reason']}；质量提前停止始终关闭。\n\n"
             f"最后保存轮数：{state['iteration']}；最后保存模型：{state['checkpoint']}。\n\n"
             f"多指标比较后保留的已验证模型：{state.get('best_checkpoint', '尚无完整评估')}。")
    except KeyboardInterrupt:
        state["status"] = "stopped"
        note(f"## 人工停止 — {stamp()}\n\n保留已完成阶段权重。")
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        note(f"## 运行异常 — {stamp()}\n\n{exc}。已完成阶段权重保留。")
        raise
    finally:
        state["finished"] = stamp()
        state.pop("child_pid", None)
        write_json(session / "state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    p = subs.add_parser("worker")
    for key in ("settings", "session", "run-name"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--resume")
    p.add_argument("--boundary", type=int, required=True)
    p.add_argument("--deadline", type=float, required=True)
    p = subs.add_parser("supervise")
    p.add_argument("--session", required=True)
    p.add_argument("--hours", type=float, default=3)
    p.add_argument("--continue-session", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.mode == "worker":
        worker(args)
    else:
        if not math.isfinite(args.hours) or args.hours <= 0:
            parser.error("hours must be finite and positive")
        supervise(args)


if __name__ == "__main__":
    main()
