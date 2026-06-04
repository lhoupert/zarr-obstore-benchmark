"""Minimal reproduction of the original notebook finding.

The original notebook (https://github.com/ofk123/dataloading-example) timed
xr.open_zarr() using %%timeit with obstore (authenticated) vs gcsfs (cached token).
This script replicates that measurement under *equal anonymous auth* to isolate whether
the open-phase slowdown persists once the credential asymmetry is removed.

Usage:
    python reproduce_notebook.py
"""
from __future__ import annotations

import timeit
import warnings

warnings.filterwarnings("ignore")

import zarr
import xarray as xr
import gcsfs
from obstore.store import GCSStore
from zarr.storage import FsspecStore, ObjectStore

BUCKET = "gcp-public-data-arco-era5"
PATH = "ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
N = 3  # repetitions (same as notebook's %%timeit default of 3 runs)


def make_obstore():
    gcsstore = GCSStore(BUCKET, prefix=PATH, skip_signature=True)
    return ObjectStore(gcsstore, read_only=True)


def make_fsspec():
    fs = gcsfs.GCSFileSystem(token="anon")
    return FsspecStore(fs, path=f"{BUCKET}/{PATH}")


def open_once(store_factory):
    store = store_factory()
    ds = xr.open_zarr(store, consolidated=False)
    ds.close()


def measure(name: str, store_factory, n: int = N) -> list[float]:
    times = []
    for i in range(n):
        t = timeit.timeit(lambda: open_once(store_factory), number=1)
        times.append(t)
        print(f"  [{name}] run {i+1}/{n}: {t:.3f}s")
    return times


if __name__ == "__main__":
    import statistics

    print(f"zarr {zarr.__version__}, obstore, gcsfs {gcsfs.__version__}")
    print(f"Dataset: gs://{BUCKET}/{PATH}")
    print(f"Measuring xr.open_zarr() — metadata + coords only, {N} repetitions each\n")

    print("=== obstore (anonymous: skip_signature=True) ===")
    obs_times = measure("obstore", make_obstore)

    print("\n=== fsspec/gcsfs (anonymous: token='anon') ===")
    fsspec_times = measure("fsspec", make_fsspec)

    obs_med = statistics.median(obs_times)
    fsspec_med = statistics.median(fsspec_times)
    ratio = obs_med / fsspec_med

    print("\n=== Results ===")
    print(f"obstore  median: {obs_med:.3f}s  (runs: {[f'{t:.3f}' for t in obs_times]})")
    print(f"fsspec   median: {fsspec_med:.3f}s  (runs: {[f'{t:.3f}' for t in fsspec_times]})")
    print(f"ratio obstore/fsspec: {ratio:.2f}x")

    if ratio > 1.2:
        print(
            "\nFinding: obstore IS slower for the open phase even under equal anonymous auth."
        )
        print("Gap is likely per-request/connection overhead, not credential resolution.")
    elif ratio < 0.85:
        print("\nFinding: obstore is FASTER for the open phase under equal anonymous auth.")
    else:
        print("\nFinding: open-phase times are roughly equal under equal anonymous auth.")
        print("The notebook gap was likely due to the credential asymmetry (G2 confound removed).")
