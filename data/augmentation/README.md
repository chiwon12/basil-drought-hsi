# Augmentation caches

In the revised pipeline, augmentation is built **inside each CV fold** to avoid
leakage (the v1 single pre-split CSV is gone). **The exact caches used in the
paper are shipped here** under `cache_{swir,vnir}/`:

```
cache_<sensor>/{ae,gan}_fold{0..4}.npz   # built from rows with fold != k only
cache_<sensor>/{ae,gan}_full.npz         # built from the whole new-train
```

Each `.npz` holds `X` (raw-reflectance features, 300/class) and `y` (int labels),
with the fold's own originals included. `src/training/run_single_experiment.py
--data ae|gan` consumes these; `--data raw` ignores them.

To regenerate them from scratch (e.g. for a different seed):

```bash
# run from the repository root
python src/augmentation/augment_with_autoencoder.py --sensor sw --build_cache
python src/augmentation/augment_with_gan.py         --sensor sw --build_cache
# repeat with --sensor vn
```
