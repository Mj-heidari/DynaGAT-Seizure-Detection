DynaGAT-Onset clean temporal pipeline
====================================

WHAT CHANGED
------------
1. The TCN now receives real ordered temporal clips [B,T,18,F].
2. The old sparse background cache is not reused. A new continuous compact v2
   cache is required so temporal context and FA/hour are scientifically valid.
3. Test patients are never used to choose a threshold. Each LOPO fold reserves
   one non-test patient group for inner validation.
4. Full validation is performed once after training, not every epoch.
5. Training resamples background clips each epoch, reducing expensive GAT work.
6. Cache tensors are compact (fp16/uint8) and mmap-backed when PyTorch supports it.

FIRST RUN
---------
1. Open config.py and check BIDS_ROOT, or set the environment variable:
      CHBMIT_BIDS_ROOT=D:\path\to\BIDS_CHB-MIT

2. Build the NEW cache. This is required once:
      python run_preprocessing.py

   Optional one-subject preprocessing smoke test:
      python run_preprocessing.py --max-subjects 1

3. After all patient caches exist, run a short training smoke test WITHOUT editing code:
      python run_training.py --max-folds 2 --epochs 2

4. If that succeeds, run the complete experiment:
      python run_training.py

OUTPUTS
-------
results/lopo_results_summary.csv
results/dynagat_onset_fold_*.pt

RTX 3060
--------
Default batch size is 24 temporal clips. If peak VRAM is comfortably low, try:
      python run_training.py --batch-size 32

If preprocessing runs out of VRAM, edit config.py and reduce:
      PREPROCESS_CHUNK_WINDOWS = 128
   to:
      PREPROCESS_CHUNK_WINDOWS = 64

IMPORTANT
---------
Do not copy old *_graphs.pt files into data_cache_v2. They skipped most background
windows and cannot provide continuous temporal evaluation.
