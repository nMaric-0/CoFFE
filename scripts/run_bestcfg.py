"""Orchestrator for the BEST-CONFIG rerun of the main experiment.

Trains every cell x seed to 1000 epochs and evaluates at epoch 1000, then
aggregates mean +/- std +/- 95% CI. Global greedy scheduler over 4 GPUs
(2 each -> 8 slots); new jobs launch as slots free; idempotent + resumable.

Usage:
    python scripts/run_bestcfg.py --stage all --dry-run
    python scripts/run_bestcfg.py --stage all
    python scripts/run_bestcfg.py --stage train --groups enhanced
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

from scripts import bestcfg_config as cfg  # noqa: E402

LOG_DIR = cfg.EXPERIMENTS_ROOT / "_bestcfg_logs"

#: front-load slow dataset
_DS_RANK = {"muufl": 0, "trento": 1, "houston": 2}


class Job:
    def __init__(self, label: str, dataset: str, argv: List[str], logfile: Path):
        self.label = label
        self.dataset = dataset
        self.argv = argv
        self.logfile = logfile


def build_jobs(phase: str, groups: Optional[List[str]], seeds: Optional[List[int]]) -> List[Job]:
    worker = "bestcfg_pretrain_worker.py" if phase == "train" else "bestcfg_eval_worker.py"
    suffix = phase
    jobs: List[Job] = []
    for group, dataset, variant, seed, name in cfg.runs(groups, seeds):
        argv = [sys.executable, str(REPO_ROOT / "scripts" / worker),
                "--group", group, "--dataset", dataset, "--variant", variant, "--seed", str(seed)]
        jobs.append(Job(name, dataset, argv, LOG_DIR / f"{name}.{suffix}.log"))
    jobs.sort(key=lambda j: (_DS_RANK.get(j.dataset, 9), j.label))
    return jobs


def run_global(jobs: List[Job], phase: str, gpus: Optional[List[str]] = None) -> List[Tuple[str, int]]:
    gpus = gpus or cfg.GPUS
    total = sum(cfg.GPU_SLOTS[g] for g in gpus)
    results: List[Tuple[str, int]] = []
    load = {g: 0 for g in gpus}
    running: List[Tuple[Job, subprocess.Popen, "object", str]] = []
    queue = list(jobs)

    def free_gpu() -> Optional[str]:
        cand = [g for g in gpus if load[g] < cfg.GPU_SLOTS[g]]
        return min(cand, key=lambda g: load[g]) if cand else None

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
            print(f"  [{phase}|{gpu}] START {job.label} "
                  f"[load {sum(load.values())}/{total}, queue {len(queue)}]", flush=True)
        time.sleep(2.0)
        still = []
        for job, proc, fh, gpu in running:
            rc = proc.poll()
            if rc is None:
                still.append((job, proc, fh, gpu)); continue
            fh.close(); load[gpu] -= 1
            print(f"  [{phase}|{gpu}] DONE  {job.label}  {'OK' if rc == 0 else f'FAIL(rc={rc})'}", flush=True)
            results.append((job.label, rc))
        running = still
    return results


def print_schedule(jobs: List[Job], phase: str, gpus: Optional[List[str]] = None):
    gpus = gpus or cfg.GPUS
    slots = {g: cfg.GPU_SLOTS[g] for g in gpus}
    print(f"\n=== {phase.upper()} schedule: {len(jobs)} jobs, "
          f"{sum(slots.values())} slots ({slots}) ===")
    for i, job in enumerate(jobs):
        print(f"  [{i + 1:3d}] {job.label}")


def summarize(results, phase) -> int:
    failed = [l for l, rc in results if rc != 0]
    print(f"\n=== {phase} summary: {len(results)} jobs, {len(failed)} failed ===")
    for l in failed:
        print(f"  FAILED: {l}")
    return len(failed)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["train", "eval", "aggregate", "all"], default="all")
    p.add_argument("--groups", nargs="+", choices=cfg.GROUPS_USED, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--gpus", nargs="+", choices=cfg.GPUS, default=None,
                   help="Subset of GPUs to use (default: all). e.g. --gpus cuda:0 cuda:2 cuda:3 to free cuda:1.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    gpus = args.gpus or cfg.GPUS

    train_jobs = build_jobs("train", args.groups, args.seeds)
    eval_jobs = build_jobs("eval", args.groups, args.seeds)

    if args.dry_run:
        if args.stage in ("train", "all"):
            print_schedule(train_jobs, "train", gpus)
        if args.stage in ("eval", "all"):
            print_schedule(eval_jobs, "eval", gpus)
        if args.stage in ("aggregate", "all"):
            print(f"\n=== AGGREGATE ===\n  $ {sys.executable} "
                  f"{REPO_ROOT / 'scripts' / 'aggregate_bestcfg.py'}")
        print("\n(dry-run: nothing launched)")
        return 0

    n_failed = 0
    if args.stage in ("train", "all"):
        n_failed += summarize(run_global(train_jobs, "train", gpus), "train")
    if args.stage in ("eval", "all"):
        n_failed += summarize(run_global(eval_jobs, "eval", gpus), "eval")
    if args.stage in ("aggregate", "all"):
        print("\n=== AGGREGATE ===", flush=True)
        n_failed += int(subprocess.call(
            [sys.executable, str(REPO_ROOT / "scripts" / "aggregate_bestcfg.py")],
            cwd=str(REPO_ROOT)) != 0)
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
