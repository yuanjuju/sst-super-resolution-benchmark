#!/usr/bin/env python3
"""Run the bounded SAFMN FFT/no-FFT pair sequentially; never resume or retry.

Launch this CPU-only manager detached. Its children use SAFMN/.venv-5090;
evaluate_safmn_fft_pair.py must be supplied in the suite directory.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


STEPS = tuple(range(5000, 30001, 5000))
EXPERIMENTS = {
    "fft": "SAFMN_DF2K_x4_pair_fft_30000",
    "nofft": "SAFMN_DF2K_x4_pair_nofft_30000",
}
PROGRESS = re.compile(r"\[epoch:\s*\d+,\s*iter:\s*([\d,]+),")
LOSS = re.compile(r"\b(l_\w+)\s*:\s*([^\s]+)")
PSNR = re.compile(r"#\s*psnr:\s*([^\s]+)", re.IGNORECASE)
ERROR = re.compile(
    r"Traceback \(most recent call last\):|\b(?:RuntimeError|ValueError|"
    r"AssertionError|OSError|ImportError|ModuleNotFoundError|TypeError|"
    r"KeyError|FileNotFoundError|MemoryError):|\bERROR\b|"
    r"Save (?:training state|model|network) error|Still cannot save"
)

CONFIG_CHECK = r'''
import copy, hashlib, json, sys
from pathlib import Path
import yaml
suite = Path(sys.argv[1])
configs = [yaml.safe_load((suite / 'configs' / f'train_{key}_30000.yml').read_text())
           for key in ('fft', 'nofft')]
for key, cfg in zip(('fft', 'nofft'), configs):
    assert cfg['name'] == f'SAFMN_DF2K_x4_pair_{key}_30000'
    assert cfg['network_g']['type'] == 'SAFMN' and cfg['model_type'] == 'SRModel'
    assert cfg['num_gpu'] == 1 and cfg['scale'] == 4
    assert cfg['train']['total_iter'] == 30000
    assert cfg['train']['ema_decay'] > 0
    assert cfg['val']['val_freq'] == cfg['logger']['save_checkpoint_freq'] == 5000
    assert cfg['path'].get('resume_state') is None
    assert cfg['path']['strict_load_g'] is True
    assert cfg['path'].get('param_key_g', 'params') == 'params'
    assert Path(cfg['path']['pretrain_network_g']).resolve() == (suite / 'common_init.pth').resolve()
    assert cfg['datasets']['val']['name'] == 'DIV2K_val100'
    metric = cfg['val']['metrics']['psnr']
    assert metric['type'] == 'calculate_psnr'
    assert metric['test_y_channel'] is False and metric['crop_border'] == 4
assert configs[0]['train']['fft_opt']['type'] == 'FFTLoss'
assert configs[0]['train']['fft_opt']['loss_weight'] > 0
assert configs[1]['train'].get('fft_opt') is None
normalized = copy.deepcopy(configs)
for cfg in normalized:
    cfg.pop('name')
    cfg['train'].pop('fft_opt', None)
assert normalized[0] == normalized[1], 'Configs differ beyond name and FFT loss'
print('CONFIG_CHECK_PASSED', flush=True)
'''

CHECKPOINT_CHECK = r'''
import hashlib, json, sys
from pathlib import Path
import torch, yaml
from basicsr.archs import build_network
torch.set_num_threads(2)
exp, config, output = map(Path, sys.argv[1:])
def finite_tree(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all().item(), 'Nonfinite checkpoint tensor'
    elif isinstance(value, dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value, (tuple, list)):
        for v in value: finite_tree(v)
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()
records = []
for step in range(5000, 30001, 5000):
    weight_path = exp / 'models' / f'net_g_{step}.pth'
    state_path = exp / 'training_states' / f'{step}.state'
    weights = torch.load(weight_path, map_location='cpu', weights_only=True)
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    assert {'params', 'params_ema'} <= weights.keys()
    assert weights['params'] and weights['params_ema']
    for key in ('params', 'params_ema'):
        assert all(isinstance(t, torch.Tensor) for t in weights[key].values())
        finite_tree(weights[key])
    assert state['iter'] == step
    assert len(state['optimizers']) == len(state['schedulers']) == 1
    finite_tree(state)
    if step == 30000:
        net = build_network(yaml.safe_load(config.read_text())['network_g'])
        net.load_state_dict(weights['params'], strict=True)
        net.load_state_dict(weights['params_ema'], strict=True)
    records.append({'iter': step, 'model': str(weight_path), 'state': str(state_path),
                    'model_sha256': digest(weight_path), 'state_sha256': digest(state_path),
                    'params_tensors': len(weights['params']),
                    'params_ema_tensors': len(weights['params_ema'])})
output.write_text(json.dumps({'status': 'passed', 'checkpoints': records}, indent=2) + '\n')
print('CHECKPOINT_CHECK_PASSED', flush=True)
'''


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(handle)


def _digest(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def scan_log(path):
    """Attach validation to actual progress, never to the `Best ... @ iter` text."""
    result = {"latest_iter": 0, "validations": {}, "errors": [], "ended": False}
    awaiting_psnr = False
    for line in path.read_text(errors="replace").splitlines():
        if ERROR.search(line):
            result["errors"].append(line[-1200:])
        match = PROGRESS.search(line)
        if match:
            result["latest_iter"] = int(match.group(1).replace(",", ""))
            for key, raw in LOSS.findall(line):
                try:
                    finite = math.isfinite(float(raw))
                except ValueError:
                    finite = False
                if not finite:
                    result["errors"].append(f"Nonfinite or invalid {key}: {raw}")
        if "End of training." in line:
            result["ended"] = True
        if "Validation DIV2K_val100" in line:
            awaiting_psnr = True
        elif awaiting_psnr and (metric := PSNR.search(line)):
            score = float(metric.group(1))
            if not math.isfinite(score):
                result["errors"].append(f"Nonfinite validation PSNR: {metric.group(1)}")
            elif not result["ended"]:
                result["validations"][str(result["latest_iter"])] = score
            awaiting_psnr = False
    return result


class Runner:
    def __init__(self, root, suite, poll_seconds):
        self.root, self.suite = root, suite
        self.project = root / "SAFMN"
        self.python = self.project / ".venv-5090/bin/python"
        self.poll_seconds = poll_seconds
        self.child = None
        self.state = {
            "status": "running", "phase": "preparing", "current_run": None,
            "completed_runs": [], "details": {}, "started_at": now(), "finished_at": None,
            "paths": {"suite": str(suite), "project": str(self.project),
                      "python": str(self.python), "runner_log": str(suite / "runner.log"),
                      "configs": {key: str(suite / f"configs/train_{key}_30000.yml") for key in EXPERIMENTS},
                      "logs": {key: str(suite / f"{key}.stdout.log") for key in EXPERIMENTS},
                      "experiments": {key: str(self.project / "experiments" / name)
                                      for key, name in EXPERIMENTS.items()}},
        }
        self.env = dict(os.environ, PYTHONPATH=str(self.project), OMP_NUM_THREADS="2",
                        OPENBLAS_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
                        MPLCONFIGDIR=str(self.project / ".cache/matplotlib"),
                        TORCH_HOME=str(self.project / ".cache/torch"),
                        CUDA_CACHE_PATH=str(self.project / ".cache/cuda"))

    def save(self, **changes):
        self.state.update(changes)
        self.state["updated_at"] = now()
        atomic_json(self.suite / "status.json", self.state)

    def log(self, message):
        with (self.suite / "runner.log").open("a") as handle:
            handle.write(f"{now()} {message}\n")

    def stop_owned_child(self):
        if self.child is None or self.child.poll() is not None:
            return
        # Only this runner's still-live, separately created process group.
        try:
            if os.getpgid(self.child.pid) != self.child.pid:
                raise RuntimeError("Unexpected child process group; refusing to signal")
            os.killpg(self.child.pid, signal.SIGTERM)
        except ProcessLookupError:
            self.child.wait()
            return
        try:
            self.child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self.child.pid, signal.SIGKILL)
            self.child.wait()

    def run_child(self, command, output_path, training=False, pid_path=None):
        with output_path.open("x") as output:
            self.child = subprocess.Popen(command, cwd=self.project, env=self.env,
                                          stdin=subprocess.DEVNULL, stdout=output,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            if pid_path:
                pid_path.write_text(str(self.child.pid) + "\n")
            self.save(child_pid=self.child.pid)
            self.log(f"Started {self.state['phase']}; pid={self.child.pid}")
            while self.child.poll() is None:
                if training:
                    details = scan_log(output_path)
                    self.save(details=details)
                    if details["errors"]:
                        raise RuntimeError(f"Training error: {details['errors'][-1]}")
                try:
                    self.child.wait(timeout=self.poll_seconds)
                except subprocess.TimeoutExpired:
                    pass
            code = self.child.returncode
            self.child = None
            self.save(child_pid=None)
            if code != 0:
                raise RuntimeError(f"Child exited with code {code}; inspect {output_path}")

    def execute(self):
        for path in (self.suite / "runner.pid", self.suite / "status.json", self.suite / "runner.log"):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing suite artifact: {path}")
        for key, name in EXPERIMENTS.items():
            if (self.project / "experiments" / name).exists():
                raise FileExistsError(f"Experiment already exists: {name}")
            if (self.suite / f"{key}.stdout.log").exists():
                raise FileExistsError(f"Training log already exists: {key}")
        preflight = json.loads((self.suite / "preflight.json").read_text())
        if preflight.get("status") != "passed":
            raise RuntimeError("preflight.json does not report status=passed")
        for path in (self.python, self.suite / "common_init.pth", self.suite / "evaluate_safmn_fft_pair.py"):
            if not path.is_file():
                raise FileNotFoundError(path)
        (self.suite / "runner.pid").write_text(str(os.getpid()) + "\n")
        self.save()
        self.log("Starting fresh, sequential FFT/no-FFT pair; 30000 steps each")
        try:
            self.run_child([str(self.python), "-c", CONFIG_CHECK, str(self.suite)],
                           self.suite / "config_check.stdout.log")
            inputs = {"common_init_sha256": sha256(self.suite / "common_init.pth"),
                      "config_sha256": {key: sha256(Path(path))
                                        for key, path in self.state["paths"]["configs"].items()}}
            if preflight.get("common_init_sha256") != inputs["common_init_sha256"]:
                raise RuntimeError("Common initializer differs from the successful preflight")
            for key, path in self.state["paths"]["configs"].items():
                if preflight.get("config_sha256", {}).get(path) != inputs["config_sha256"][key]:
                    raise RuntimeError(f"Configuration differs from the successful preflight: {key}")
            self.save(inputs=inputs)
            for key, name in EXPERIMENTS.items():
                exp = self.project / "experiments" / name
                if exp.exists():
                    raise FileExistsError(f"Experiment appeared before launch: {exp}")
                config = self.suite / f"configs/train_{key}_30000.yml"
                if sha256(config) != inputs["config_sha256"][key] or sha256(self.suite / "common_init.pth") != inputs["common_init_sha256"]:
                    raise RuntimeError("Config or common initialization changed during the queue")
                logfile = self.suite / f"{key}.stdout.log"
                self.save(phase="training", current_run=key, details={})
                self.run_child([str(self.python), "-u", "basicsr/train.py", "-opt", str(config)],
                               logfile, training=True, pid_path=self.suite / f"{key}.pid")
                details = scan_log(logfile)
                if details["errors"] or not details["ended"] or details["latest_iter"] != 30000:
                    raise RuntimeError(f"Incomplete or invalid training log: {details}")
                missing_scores = [step for step in STEPS if str(step) not in details["validations"]]
                if missing_scores:
                    raise RuntimeError(f"Missing scheduled EMA validation: {missing_scores}")
                for step in STEPS:
                    for path in (exp / f"models/net_g_{step}.pth", exp / f"training_states/{step}.state"):
                        if not path.is_file() or path.stat().st_size == 0:
                            raise FileNotFoundError(f"Missing or empty checkpoint: {path}")
                self.save(phase="verifying", details=details)
                report_path = self.suite / f"{key}.checkpoint_verification.json"
                self.run_child([str(self.python), "-c", CHECKPOINT_CHECK, str(exp), str(config), str(report_path)],
                               self.suite / f"{key}.verification.stdout.log")
                verification = json.loads(report_path.read_text())
                if verification.get("status") != "passed":
                    raise RuntimeError(f"Checkpoint verification failed: {key}")
                self.state["completed_runs"].append({"run": key, "completed_at": now(),
                    "validations": details["validations"], "verification": str(report_path),
                    "final_psnr_rgb_ema": details["validations"]["30000"]})
                self.save()
                self.log(f"Verified {key}: 30000 steps, EMA RGB PSNR={details['validations']['30000']}")
            self.save(phase="evaluating", current_run=None, details={})
            self.run_child([str(self.python), "-u", str(self.suite / "evaluate_safmn_fft_pair.py"),
                            "--root", str(self.root), "--suite", str(self.suite)],
                           self.suite / "evaluation.stdout.log")
            evaluation = json.loads((self.suite / "results/evaluation.json").read_text())
            if evaluation.get("completed") is not True:
                raise RuntimeError("Evaluation did not report successful completion")
            for name in ("training_metrics.csv", "set5_per_image.csv", "set5_mean.csv", "training_curve.png"):
                path = self.suite / "results" / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing evaluation artifact: {path}")
            if len(list((self.suite / "results/comparisons").glob("*.png"))) != 5:
                raise RuntimeError("Expected five Set5 comparison figures")
            self.save(status="completed", phase="completed", finished_at=now())
            self.log("Both runs and paired evaluation completed; shut down cloud instance separately")
        except BaseException as error:
            self.stop_owned_child()
            self.save(status="failed", phase="failed", child_pid=None, finished_at=now(),
                      error=f"{type(error).__name__}: {error}")
            self.log(f"FAILED: {type(error).__name__}: {error}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/fsas/sr-repro"))
    parser.add_argument("--suite", type=Path, help="Default: ROOT/safmn_fft_pair_30k")
    parser.add_argument("--poll-seconds", type=float, default=20)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    root = args.root.resolve()
    suite = (args.suite or root / "safmn_fft_pair_30k").resolve()
    if not suite.is_dir():
        parser.error(f"Suite directory must already exist: {suite}")
    with (suite / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another runner holds this suite's lock")
        def interrupted(signum, _frame):
            raise RuntimeError(f"Runner interrupted by signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        Runner(root, suite, args.poll_seconds).execute()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Runner failed: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
