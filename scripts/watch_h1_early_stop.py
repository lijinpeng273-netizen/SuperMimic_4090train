"""Attach the quality gate to a supervisor that was started before the gate existed."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import time

from supermimic.training.early_stop import quality_goal


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def supervisor_alive(pid, session):
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        tokens = [v.decode() for v in args if v]
        status = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        return (status != 'Z' and 'scripts/supervise_h1.py' in tokens
                and 'supervise' in tokens and str(session) in tokens)
    except (OSError, UnicodeError, IndexError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    session = args.session.resolve()
    if not session.is_relative_to(root):
        raise ValueError('All outputs must remain in the H1 project')
    state = json.loads((session/'state.json').read_text())
    pid, deadline = state['pid'], state['deadline_unix']
    report = Path(state['report']).resolve()
    if not report.is_relative_to(root):
        raise ValueError('Report must remain in the H1 project')
    observer_file = session/'early_stop_watcher.json'
    write_json(observer_file, {'pid': os.getpid(), 'supervisor_pid': pid, 'status': 'watching'})
    print(f'Quality gate attached to supervisor {pid}', flush=True)
    last_version = None
    while time.time() < deadline + 120 and supervisor_alive(pid, session):
        path = session/'history.json'
        if not path.exists():
            time.sleep(5)
            continue
        policy_path = session/'early_stop_policy.json'
        version = (path.stat().st_mtime_ns, policy_path.stat().st_mtime_ns)
        if version != last_version:
            history = json.loads(path.read_text())
            policy = json.loads(policy_path.read_text())
            reached, reason = quality_goal(history, policy)
            last_version = version
            print(reason, flush=True)
            if reached:
                # state.json points to the saved, evaluated stage before evaluation starts.
                state = json.loads((session/'state.json').read_text())
                checkpoint = session/f"stage_{history[-1]['iteration']:06d}"/'ckpt_latest.pt'
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                event = {'reason': reason, 'iteration': history[-1]['iteration'],
                         'checkpoint': str(checkpoint), 'policy': policy,
                         'time': dt.datetime.now().astimezone().isoformat(timespec='seconds')}
                write_json(session/'early_stop_result.json', event)
                if supervisor_alive(pid, session):
                    os.kill(pid, signal.SIGTERM)
                for _ in range(30):
                    if not supervisor_alive(pid, session):
                        break
                    time.sleep(1)
                if supervisor_alive(pid, session):
                    raise RuntimeError('Supervisor did not finish its graceful stop; weights retained')
                state = json.loads((session/'state.json').read_text())
                state.update(status='early_stopped', stop_reason=reason,
                             checkpoint=str(checkpoint), iteration=event['iteration'])
                write_json(session/'state.json', state)
                with report.open('a') as handle:
                    handle.write(f"\n## 提前停止：效果达标 — {event['time']}\n\n{reason}\n\n"
                                 f"保存权重：`{checkpoint}`。旧监管程序的通用信号退出已被标记为本次效果达标提前停止。\n")
                write_json(observer_file, {'pid': os.getpid(), 'status': 'early_stopped', **event})
                print('Early stop completed; evaluated checkpoint retained.', flush=True)
                return
        time.sleep(5)
    write_json(observer_file, {'pid': os.getpid(), 'status': 'finished_without_early_stop'})


if __name__ == '__main__':
    main()
