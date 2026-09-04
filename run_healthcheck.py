"""
Pre-flight check. Run this before committing to a multi-hour LOPO sweep.

    python run_healthcheck.py

Verifies the environment, the BIDS root, the built caches, a model
forward/backward on the real cache shapes, and prints a measured throughput
estimate for the full run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C

_problems: list[str] = []


def line(ok: bool, text: str, detail: str = "") -> None:
    print(("  [ok]   " if ok else "  [FAIL] ") + text + (f"  {detail}" if detail else ""))
    if not ok:
        _problems.append(text)


def main() -> int:
    print("=" * 78)
    print("DynaGAT health check")
    print("=" * 78)

    print("\nEnvironment")
    line(sys.version_info >= (3, 9), f"python {sys.version.split()[0]}")
    line(True, f"torch {torch.__version__}")
    cuda = torch.cuda.is_available()
    line(cuda, "CUDA available", torch.cuda.get_device_name(0) if cuda else "training will be very slow on CPU")
    if cuda:
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        line(total >= 3.5, f"GPU memory {total:.1f} GB", "reduce --batch-size if under 6 GB")
    for mod in ("mne", "scipy", "pandas", "numpy", "tqdm", "matplotlib"):
        try:
            __import__(mod)
            line(True, f"package {mod}")
        except ImportError:
            line(False, f"package {mod} missing", "pip install -r requirements.txt")

    print("\nDataset")
    line(C.BIDS_ROOT.exists(), f"BIDS root {C.BIDS_ROOT}",
         "" if C.BIDS_ROOT.exists() else "set CHBMIT_BIDS_ROOT")
    if C.BIDS_ROOT.exists():
        subs = sorted(d for d in C.BIDS_ROOT.iterdir() if d.is_dir() and d.name.startswith("sub-"))
        edf = sum(1 for _ in C.BIDS_ROOT.rglob("*.edf"))
        line(len(subs) >= 20, f"{len(subs)} subject directories")
        line(edf > 600, f"{edf} EDF files found")

    print("\nCaches")
    caches = sorted(C.PROCESSED_DATA_DIR.glob("sub-*_v4.pt"))
    line(len(caches) > 0, f"{len(caches)} subject cache(s) in {C.PROCESSED_DATA_DIR}",
         "" if caches else "run: python -m dataset.preprocess")
    total_windows = 0
    total_seizures = 0
    if caches:
        from dataset.sequence_dataset import load_cache
        bad = []
        for p in caches:
            try:
                c = load_cache(p)
                total_windows += int(c["total_windows"])
                total_seizures += int(c["total_seizures"])
                del c
            except Exception as exc:
                bad.append(f"{p.name}: {exc}")
        line(not bad, "all caches validate", "; ".join(bad[:3]))
        line(total_windows > 0, f"{total_windows:,} windows, {total_seizures} seizures")

    print("\nModel")
    try:
        from models.dynagat import DynaGAT
        dev = torch.device("cuda" if cuda else "cpu")
        model = DynaGAT().to(dev)
        n = sum(p.numel() for p in model.parameters())
        line(True, f"model builds", f"{n:,} parameters")
        b, t, k = C.BATCH_SIZE, C.SEQUENCE_LENGTH, C.TOP_K_CAUSAL
        x = torch.randn(b, t, 18, C.NODE_FEATURE_DIM, device=dev)
        ind = torch.randint(0, 18, (b, t, 18, k), device=dev)
        inw = torch.rand(b, t, 18, k, device=dev)
        vm = torch.ones(b, t, dtype=torch.bool, device=dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_iter = 6
        for _ in range(n_iter):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=cuda):
                out = model(x, ind, inw, ind, inw, vm)
                loss = out.mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if cuda:
            torch.cuda.synchronize()
        per_step = (time.perf_counter() - t0) / n_iter
        line(True, f"train step {per_step*1000:.0f} ms at batch {b}")
        if cuda:
            line(True, f"peak VRAM {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        # rough projection
        clips_per_epoch = 25000
        steps = clips_per_epoch / b
        fold_min = (steps * per_step * C.EPOCHS) / 60.0
        eval_min = (total_windows / max(1, len(caches))) * 6 / (b * t) * per_step * 0.4 / 60.0
        line(True, f"projected ~{fold_min + eval_min:.1f} min/fold",
             f"~{(fold_min + eval_min) * max(1, len(caches))/60:.1f} h for a full LOPO sweep")
    except Exception as exc:
        line(False, f"model check failed: {exc}")

    print("\n" + "=" * 78)
    if _problems:
        print(f"{len(_problems)} problem(s) to resolve before the full run:")
        for p in _problems:
            print(f"  - {p}")
        return 1
    print("ready. next:  python run_lopo.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
