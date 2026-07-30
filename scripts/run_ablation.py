"""Orchestrator for the OFAT hyperparameter ablation.

Trains every variant x seed to 1000 epochs and evaluates each at epoch 800 and
1000, then aggregates a mean +/- std +/- 95% CI report. Uses the same global
greedy GPU scheduler as the significance harness: one queue, <= 2 runs per GPU,
new jobs launch the instant a slot frees. 4 GPUs -> 8 concurrent.

Each run is its own subprocess; both stages are idempotent so the orchestrator
is freely resumable.

Usage:
    python scripts/run_ablation.py --stage all --dry-run
    python scripts/run_ablation.py --stage all
    python scripts/run_ablation.py --stage train --slugs baseline center_off
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import ablation_config as cfg  # noqa: E402

LOG_DIR = cfg.EXPERIMENTS_ROOT / "_ablation_logs"


class Job:
    def __init__(self, label: str, argv: List[str], logfile: Path):
        self.label = label
        self.argv = argv  # without --device
        self.logfile = logfile


def build_jobs(stage: str, slugs: Optional[List[str]], seeds: Optional[List[int]]) -> List[Job]:
    jobs: List[Job] = []
    for slug, seed, name in cfg.runs(slugs, seeds):
        if stage == "train":
            argv = [
                sys.executable, str(REPO_ROOT / "scripts" / "ablation_pretrain_worker.py"),
                "--slug", slug, "--seed", str(seed),
            ]
            jobs.append(Job(name, argv, LOG_DIR / f"{name}.train.log"))
        else:  # eval: one job per epoch
            for epoch in cfg.EVAL_EPOCHS:
                argv = [
                    sys.executable, str(REPO_ROOT / "scripts" / "ablation_eval_worker.py"),
                    "--slug", slug, "--seed", str(seed), "--epoch", str(epoch),
                ]
                jobs.append(Job(f"{name}@e{epoch}", argv, LOG_DIR / f"{name}.eval_e{epoch}.log"))
    jobs.sort(key=lambda j: j.label)
    return jobs


def run_global(jobs: List[Job], stage: str) -> List[Tuple[str, int]]:
    """Global queue with <= MAX_PARALLEL_PER_GPU per GPU."""
    results: List[Tuple[str, int]] = []
    load = {gpu: 0 for gpu in cfg.GPUS}
    running: List[Tuple[Job, subprocess.Popen, "object", str]] = []
    queue = list(jobs)
    n_slots = len(cfg.GPUS) * cfg.MAX_PARALLEL_PER_GPU

    def free_gpu() -> Optional[str]:
        candidates = [g for g in cfg.GPUS if load[g] < cfg.MAX_PARALLEL_PER_GPU]
        return min(candidates, key=lambda g: load[g]) if candidates else None

    while queue or running:
        while queue:
            gpu = free_gpu()
            if gpu is None:
                break
            job = queue.pop(0)
            job.logfile.parent.mkdir(parents=True, exist_ok=True)
            fh = open(job.logfile, "w")
            proc = subprocess.Popen(job.argv + ["--device", gpu], stdout=fh,
                                    stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
            load[gpu] += 1
            running.append((job, proc, fh, gpu))
            print(f"  [{stage}|{gpu}] START {job.label} "
                  f"[load {sum(load.values())}/{n_slots}, queue {len(queue)}]", flush=True)

        time.sleep(2.0)

        still = []
        for job, proc, fh, gpu in running:
            rc = proc.poll()
            if rc is None:
                still.append((job, proc, fh, gpu))
                continue
            fh.close()
            load[gpu] -= 1
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  [{stage}|{gpu}] DONE  {job.label}  {status}", flush=True)
            results.append((job.label, rc))
        running = still

    return results


def print_schedule(jobs: List[Job], stage: str):
    n_slots = len(cfg.GPUS) * cfg.MAX_PARALLEL_PER_GPU
    print(f"\n=== {stage.upper()} schedule: {len(jobs)} jobs, global queue, "
          f"{n_slots} slots ({cfg.MAX_PARALLEL_PER_GPU}/GPU on {cfg.GPUS}) ===")
    for i, job in enumerate(jobs):
        print(f"  [{i + 1:3d}] {job.label}")
        print(f"        $ {' '.join(job.argv)} --device <auto>")


def summarize(results: List[Tuple[str, int]], stage: str) -> int:
    failed = [lbl for lbl, rc in results if rc != 0]
    print(f"\n=== {stage} summary: {len(results)} jobs, {len(failed)} failed ===")
    for lbl in failed:
        print(f"  FAILED: {lbl}")
    return len(failed)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["train", "eval", "aggregate", "all"], default="all")
    p.add_argument("--slugs", nargs="+", choices=cfg.SLUGS, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    train_jobs = build_jobs("train", args.slugs, args.seeds)
    eval_jobs = build_jobs("eval", args.slugs, args.seeds)

    if args.dry_run:
        if args.stage in ("train", "all"):
            print_schedule(train_jobs, "train")
        if args.stage in ("eval", "all"):
            print_schedule(eval_jobs, "eval")
        if args.stage in ("aggregate", "all"):
            print(f"\n=== AGGREGATE ===\n  $ {sys.executable} "
                  f"{REPO_ROOT / 'scripts' / 'aggregate_ablation.py'}")
        print("\n(dry-run: nothing launched)")
        return 0

    n_failed = 0
    if args.stage in ("train", "all"):
        n_failed += summarize(run_global(train_jobs, "train"), "train")
    if args.stage in ("eval", "all"):
        n_failed += summarize(run_global(eval_jobs, "eval"), "eval")
    if args.stage in ("aggregate", "all"):
        print("\n=== AGGREGATE ===", flush=True)
        rc = subprocess.call([sys.executable, str(REPO_ROOT / "scripts" / "aggregate_ablation.py")],
                             cwd=str(REPO_ROOT))
        n_failed += int(rc != 0)

    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
