# zarr-obstore-benchmark

Reproducible benchmark comparing **obstore** vs **fsspec/gcsfs** as zarr storage backends for
remote Zarr reads on GCS.

Motivated by [a thread](https://github.com/zarr-developers/zarr-python/issues) claiming obstore
+ zarr is ~40–50% slower than fsspec + zarr when opening a remote ERA5 Zarr store. This repo
investigates whether the slowdown is real, what causes it, and whether it affects actual data
reads or only metadata/open operations.

## What we measure

Three distinct phases, separated so they can be compared independently:

| Phase | Description |
|-------|-------------|
| **store-init** | Time to construct `ObjectStore` or `FsspecStore` |
| **open (G2)** | `xr.open_zarr()` — metadata + coordinate reads only (no bulk data) |
| **data-read (G1)** | `.isel(time=slice(0,100)).load()` on `2m_temperature` — 100 × 4.2 MB chunks ≈ 420 MB |

Key controls that fix the original notebook's confounds:

- **Equal anonymous auth**: obstore uses `skip_signature=True`, gcsfs uses `token="anon"` — no
  token-fetch asymmetry.
- **Fresh store per iteration**: no client-side xarray/fsspec cache leaks across timed runs.
- **Concurrency sweep**: `async.concurrency` ∈ {10, 32, 128, 256} via
  `zarr.config.set({"async.concurrency": N})`.
- **Equal-work check**: `np.array_equal` on the slab read by both backends.
- **Stats**: N=5 repetitions; median ± IQR reported (not mean).

## Dataset

Public GCS bucket, no credentials required:

```
gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
```

Variable used: `2m_temperature` — shape `(1323648, 721, 1440)`, chunks `(1, 721, 1440)` ≈ 4.2 MB.

## Quick start

```bash
git clone https://github.com/lhoupert/zarr-obstore-benchmark
cd zarr-obstore-benchmark
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
# or: uv pip install --python .venv/bin/python "zarr==3.2.1" obstore gcsfs xarray numpy

# Run full sweep (both adapters, all concurrency levels):
.venv/bin/python bench.py --adapter obstore --concurrency 10 32 128 256 --n 5
.venv/bin/python bench.py --adapter fsspec  --concurrency 10 32 128 256 --n 5

# Quick reproduction of the notebook's open-phase measurement:
.venv/bin/python reproduce_notebook.py
```

Results are written to `results/`.

## For zarr-python maintainers

To test against a local zarr-python checkout:
```bash
uv pip install --python .venv/bin/python -e /path/to/zarr-python
```

## Findings

See [FINDINGS.md](FINDINGS.md) for the full analysis and measured numbers.

## Cloud support

Currently implemented: **GCS** (`gcp-public-data-arco-era5`).  
S3 and Azure can be added behind the `--cloud` selector in `bench.py` without other changes.
