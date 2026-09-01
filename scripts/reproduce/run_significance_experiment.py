"""Orchestrator for the multi-seed statistical-significance experiment.

Drives training + epoch-700 evaluation of every selected cell (group x dataset x
variant) under 5 seeds, then aggregates a mean +/- std +/- 95% CI report.

Global greedy GPU scheduler: one queue of all jobs, each GPU in GPUS holds at
most MAX_PARALLEL_PER_GPU runs. Whenever any GPU has a free slot and the queue
is non-empty, the next job launches on that GPU (the orchestrator injects
--device). So up to len(GPUS)*MAX_PARALLEL_PER_GPU run concurrently and no GPU
idles while another is backed up. The queue is ordered slow-dataset-first to
front-load the bottleneck.

Each run is its own subprocess (worker), so one crash never takes down the rest.
Both stages are idempotent -- finished runs are skipped -- so the orchestrator
is freely resumable.

Usage:
    python scripts/reproduce/run_significance_experiment.py --stage all --dry-run
    python scripts/reproduce/run_significance_experiment.py --stage all \
        --groups hsi_only enhanced_mae mft_mae mft_spatial
    python scripts/reproduce/run_significance_experiment.py --stage train --groups mft_mae
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reproduce import sig_significance_config as cfg  # noqa: E402

LOG_DIR = cfg.EXPERIMENTS_ROOT / "_significance_logs"


class Job:
    """A single worker invocation; device is assigned at launch time."""

    def __init__(self, label: str, dataset: str, argv: List[str], logfile: Path):
        self.label = label
        self.dataset = dataset
        self.argv = argv  # without --device
        self.logfile = logfile


def _dataset_rank(dataset: str) -> int:
    return cfg.DATASET_COST_ORDER.index(dataset) if dataset in cfg.DATASET_COST_ORDER else 99


def build_jobs(stage: str, groups: Optional[List[str]], seeds: Optional[List[int]]) -> List[Job]:
    """stage is 'train' or 'eval'. Returns the ordered job queue."""
    worker = "sig_pretrain_worker.py" if stage == "train" else "sig_eval_worker.py"
    suffix = "train" if stage == "train" else "eval"
    jobs: List[Job] = []
    for group, dataset, variant, seed, name in cfg.runs(groups, seeds):
        argv = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "reproduce" / worker),
            "--group", group,
            "--dataset", dataset,
            "--variant", variant,
            "--seed", str(seed),
        ]
        jobs.append(Job(name, dataset, argv, LOG_DIR / f"{name}.{suffix}.log"))
    # Slow dataset first, then stable by label.
    jobs.sort(key=lambda j: (_dataset_rank(j.dataset), j.label))
    return jobs


def run_global(jobs: List[Job], stage: str) -> List[Tuple[str, int]]:
    """Run jobs through a global queue with <= MAX_PARALLEL_PER_GPU per GPU."""
    results: List[Tuple[str, int]] = []
    load = {gpu: 0 for gpu in cfg.GPUS}
    running: List[Tuple[Job, subprocess.Popen, "object", str]] = []
    queue = list(jobs)

    def free_gpu() -> Optional[str]:
        # pick the least-loaded GPU that still has a free slot
        candidates = [g for g in cfg.GPUS if load[g] < cfg.MAX_PARALLEL_PER_GPU]
        if not candidates:
            return None
        return min(candidates, key=lambda g: load[g])

    while queue or running:
        # fill free slots
        while queue:
            gpu = free_gpu()
            if gpu is None:
                break
            job = queue.pop(0)
            job.logfile.parent.mkdir(parents=True, exist_ok=True)
            fh = open(job.logfile, "w")
            argv = job.argv + ["--device", gpu]
            proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
            load[gpu] += 1
            running.append((job, proc, fh, gpu))
            print(f"  [{stage}|{gpu}] START {job.label} "
                  f"(log: {job.logfile.relative_to(REPO_ROOT)}) "
                  f"[load {sum(load.values())}/{len(cfg.GPUS) * cfg.MAX_PARALLEL_PER_GPU}, "
                  f"queue {len(queue)}]", flush=True)

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
    p.add_argument("--groups", nargs="+", choices=list(cfg.GROUPS), default=None,
                   help="Groups to run (default: all).")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Seeds to run (default: all 5).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the full queue without launching anything.")
    args = p.parse_args()

    train_jobs = build_jobs("train", args.groups, args.seeds)
    eval_jobs = build_jobs("eval", args.groups, args.seeds)

    if args.dry_run:
        if args.stage in ("train", "all"):
            print_schedule(train_jobs, "train")
        if args.stage in ("eval", "all"):
            print_schedule(eval_jobs, "eval")
        if args.stage in ("aggregate", "all"):
            print(f"\n=== AGGREGATE ===\n  $ {sys.executable} "
                  f"{REPO_ROOT / 'scripts' / 'reports' / 'aggregate_significance.py'}")
        print("\n(dry-run: nothing launched)")
        return 0

    n_failed = 0
    if args.stage in ("train", "all"):
        n_failed += summarize(run_global(train_jobs, "train"), "train")
    if args.stage in ("eval", "all"):
        n_failed += summarize(run_global(eval_jobs, "eval"), "eval")
    if args.stage in ("aggregate", "all"):
        print("\n=== AGGREGATE ===", flush=True)
        agg = [sys.executable, str(REPO_ROOT / "scripts" / "reports" / "aggregate_significance.py")]
        if args.groups:
            agg += ["--groups", *args.groups]
        rc = subprocess.call(agg, cwd=str(REPO_ROOT))
        n_failed += int(rc != 0)

    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
