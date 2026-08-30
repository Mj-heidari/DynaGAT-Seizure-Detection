from __future__ import annotations

import sys

import pandas as pd
import torch

from config import (
    NODE_FEATURE_DIM,
    NUM_NODES,
    PREPROCESSING_TAG,
    PROCESSED_DATA_DIR,
    TOP_K_DYNAMIC,
)
from models.dynagat_model import DynaGATOnsetModel


EXPECTED_EDF = 686
EXPECTED_SEIZURES = 198


def main() -> None:
    print(f"[*] Python: {sys.version.split()[0]}")
    print(f"[*] PyTorch: {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    props = torch.cuda.get_device_properties(0)
    print(f"[*] GPU: {props.name} | {props.total_memory / (1024 ** 3):.2f} GB")

    manifest_path = PROCESSED_DATA_DIR / "preprocessing_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    edf = int(manifest["edf_files"].sum())
    valid = int(manifest["valid_recordings"].sum())
    seizures = int(manifest["seizures"].sum())
    if edf != EXPECTED_EDF or valid != EXPECTED_EDF or seizures != EXPECTED_SEIZURES:
        raise RuntimeError(
            f"Dataset QA failed: EDF={edf}, valid={valid}, seizures={seizures}"
        )
    if "cache_version" in manifest.columns and not (manifest["cache_version"] == 3).all():
        raise RuntimeError("Unexpected cache version in preprocessing manifest")
    if "feature_dim" in manifest.columns and not (manifest["feature_dim"] == NODE_FEATURE_DIM).all():
        raise RuntimeError("Unexpected feature dimension in preprocessing manifest")

    caches = sorted(PROCESSED_DATA_DIR.glob("*_temporal_graphs.pt"))
    if len(caches) != len(manifest):
        raise RuntimeError(f"Cache count mismatch: {len(caches)} files for {len(manifest)} subjects")

    device = torch.device("cuda")
    model = DynaGATOnsetModel().to(device).eval()
    x = torch.zeros((1, 2, NUM_NODES, NODE_FEATURE_DIM), device=device)
    dst = torch.zeros((1, 2, NUM_NODES, TOP_K_DYNAMIC), dtype=torch.long, device=device)
    weight = torch.ones((1, 2, NUM_NODES, TOP_K_DYNAMIC), device=device)
    valid_mask = torch.ones((1, 2), dtype=torch.bool, device=device)
    with torch.inference_mode():
        logits = model(x, dst, weight, valid_mask=valid_mask)
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise RuntimeError(f"Model forward check failed: shape={tuple(logits.shape)}")

    print(f"[*] Preprocessing tag: {PREPROCESSING_TAG}")
    print(f"[*] Dataset QA: EDF={edf}, valid={valid}, seizures={seizures}")
    print(f"[*] Subject caches: {len(caches)}")
    print("[+] Health check passed")


if __name__ == "__main__":
    main()
