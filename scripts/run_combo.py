"""Orchestrator for the COMBINATION study (staged, GPUs cuda:2,3).

Stages (informative-first): anchor -> pairs -> triples -> l2o. Run one stage at
a time, review, then launch the next. `--stage-group all` runs every stage in
one deduped queue.

Global greedy scheduler: one queue, <= 2 runs per GPU on cuda:2,3 (4 slots);
new jobs launch as slots free. Each run is a subprocess; idempotent + resumable.

Usage:
    python scripts/run_combo.py --stage-group anchor --phase all --dry-run
    python scripts/run_combo.py --stage-group anchor --phase all
    python scripts/run_combo.py --stage-group pairs --phase all
    python scripts/run_combo.py --stage-group all --phase aggregate
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

from scripts import combo_config as cfg  # noqa: E402

LOG_DIR = cfg.EXPERIMENTS_ROOT / "_combo_logs"


class Job:
    def __init__(self, label: str, argv: List[str], logfile: Path):
        self.label = label
        self.argv = argv
        self.logfile = logfile


def _stages(stage_group: str) -> List[str]:
    return cfg.STAGE_ORDER if stage_group == "all" else [stage_group]


def build_jobs(phase: str, stage_group: str, seeds: Optional[List[int]]) -> List[Job]:
    stages = _stages(stage_group)
    jobs: List[Job] = []
    for factors, seed, name in cfg.runs(stages, seeds):
        fcsv = ",".join(factors)
        if phase == "train":
            argv = [sys.executable, str(REPO_ROOT / "scripts" / "combo_pretrain_worker.py"),
                    "--factors", fcsv, "--seed", str(seed)]
            jobs.append(Job(name, argv, LOG_DIR / f"{name}.train.log"))
        else:
            for epoch in cfg.EVAL_EPOCHS:
                argv = [sys.executable, str(REPO_ROOT / "scripts" / "combo_eval_worker.py"),
                        "--factors", fcsv, "--seed", str(seed), "--epoch", str(epoch)]
                jobs.append(Job(f"{name}@e{epoch}", argv, LOG_DIR / f"{name}.eval_e{epoch}.log"))
    jobs.sort(key=lambda j: j.label)
    return jobs


def run_global(jobs: List[Job], phase: str) -> List[Tuple[str, int]]:
    results: List[Tuple[str, int]] = []
    load = {gpu: 0 for gpu in cfg.GPUS}
    running: List[Tuple[Job, subprocess.Popen, "object", str]] = []
    queue = list(jobs)
    n_slots = cfg.TOTAL_SLOTS

    def free_gpu() -> Optional[str]:
        # least-loaded GPU that still has a free slot (per-GPU cap)
        cand = [g for g in cfg.GPUS if load[g] < cfg.GPU_SLOTS[g]]
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
                  f"[load {sum(load.values())}/{n_slots}, queue {len(queue)}]", flush=True)
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


def print_schedule(jobs: List[Job], phase: str, stage_group: str):
    combos = cfg.all_combos(_stages(stage_group))
    print(f"\n=== {phase.upper()} [{stage_group}]: {len(combos)} combos, {len(jobs)} jobs, "
          f"{cfg.TOTAL_SLOTS} slots ({cfg.GPU_SLOTS}) ===")
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
    p.add_argument("--stage-group", choices=cfg.STAGE_ORDER + ["all"], default="anchor")
    p.add_argument("--phase", choices=["train", "eval", "aggregate", "all"], default="all")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    train_jobs = build_jobs("train", args.stage_group, args.seeds)
    eval_jobs = build_jobs("eval", args.stage_group, args.seeds)

    if args.dry_run:
        if args.phase in ("train", "all"):
            print_schedule(train_jobs, "train", args.stage_group)
        if args.phase in ("eval", "all"):
            print_schedule(eval_jobs, "eval", args.stage_group)
        if args.phase in ("aggregate", "all"):
            print(f"\n=== AGGREGATE ===\n  $ {sys.executable} "
                  f"{REPO_ROOT / 'scripts' / 'aggregate_combo.py'}")
        print("\n(dry-run: nothing launched)")
        return 0

    n_failed = 0
    if args.phase in ("train", "all"):
        n_failed += summarize(run_global(train_jobs, "train"), "train")
    if args.phase in ("eval", "all"):
        n_failed += summarize(run_global(eval_jobs, "eval"), "eval")
    if args.phase in ("aggregate", "all"):
        print("\n=== AGGREGATE ===", flush=True)
        n_failed += int(subprocess.call(
            [sys.executable, str(REPO_ROOT / "scripts" / "aggregate_combo.py")],
            cwd=str(REPO_ROOT)) != 0)
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
