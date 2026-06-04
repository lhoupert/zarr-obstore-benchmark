"""
test_store_reuse.py — Does reusing the same store object warm up obstore's connection pool?

Compares two patterns for each backend:
  Pattern A (fresh): new store created before each open_zarr call  ← current worst-case
  Pattern B (reused): store created once, open_zarr called N times  ← workaround

If obstore's Rust HTTP client pools connections within a single GCSStore instance,
Pattern B should show a warmup curve (run 1 slow, runs 2-N faster), closing the gap
with fsspec.

Usage:
    python test_store_reuse.py
Results written to results/<timestamp>_store_reuse.csv
"""
from __future__ import annotations

import csv
import statistics
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import xarray as xr
import gcsfs
from obstore.store import GCSStore
from zarr.storage import FsspecStore, ObjectStore

BUCKET = "gcp-public-data-arco-era5"
PATH = "ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
N = 5


def make_obstore() -> ObjectStore:
    return ObjectStore(GCSStore(BUCKET, prefix=PATH, skip_signature=True), read_only=True)


def make_fsspec() -> FsspecStore:
    fs = gcsfs.GCSFileSystem(token="anon")
    return FsspecStore(fs, path=f"{BUCKET}/{PATH}")


def open_once(store) -> float:
    t0 = time.perf_counter()
    ds = xr.open_zarr(store, consolidated=True)
    elapsed = time.perf_counter() - t0
    ds.close()
    return elapsed


def measure_fresh(name: str, factory, n: int) -> list[float]:
    """Pattern A: new store before every call."""
    times = []
    for i in range(n):
        t = open_once(factory())
        times.append(t)
        print(f"  {name:8s} fresh   run {i+1}/{n}: {t:.3f}s")
    return times


def measure_reused(name: str, factory, n: int) -> list[float]:
    """Pattern B: store created once, reused for every call."""
    store = factory()
    times = []
    for i in range(n):
        t = open_once(store)
        times.append(t)
        print(f"  {name:8s} reused  run {i+1}/{n}: {t:.3f}s")
    return times


def save_csv(rows: list[dict], out_dir: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"{ts}_store_reuse.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backend", "pattern", "run", "time_s"])
        w.writeheader()
        w.writerows(rows)
    return path


def print_summary(results: dict[tuple[str, str], list[float]]) -> None:
    print("\n" + "=" * 65)
    print(f"{'backend':8s} {'pattern':8s} {'run1':>7s} {'med(2-N)':>9s} {'improvement':>12s}")
    print("-" * 65)
    for (name, pattern), times in results.items():
        cold = times[0]
        warm_med = statistics.median(times[1:]) if len(times) > 1 else cold
        speedup = cold / warm_med if warm_med > 0 else 1.0
        print(f"{name:8s} {pattern:8s} {cold:7.3f}s {warm_med:9.3f}s {speedup:11.2f}x")

    print("\nReuse speedup (fresh-median / reused-warm-median):")
    for name in ("obstore", "fsspec"):
        fresh = results.get((name, "fresh"), [])
        reused = results.get((name, "reused"), [])
        if fresh and reused:
            fresh_med = statistics.median(fresh)
            reused_warm = statistics.median(reused[1:]) if len(reused) > 1 else reused[0]
            ratio = fresh_med / reused_warm
            flag = "<-- reuse helps" if ratio > 1.1 else "  ~no difference"
            print(f"  {name}: {ratio:.2f}x  {flag}")


if __name__ == "__main__":
    import obstore, zarr
    print(f"zarr {zarr.__version__}  obstore {obstore.__version__}  gcsfs {gcsfs.__version__}")
    print(f"Dataset: gs://{BUCKET}/{PATH}  N={N} reps\n")

    results: dict[tuple[str, str], list[float]] = {}
    rows: list[dict] = []

    for name, factory in [("obstore", make_obstore), ("fsspec", make_fsspec)]:
        print(f"=== {name} — Pattern A: fresh store per call ===")
        times_fresh = measure_fresh(name, factory, N)
        results[(name, "fresh")] = times_fresh
        for i, t in enumerate(times_fresh):
            rows.append({"backend": name, "pattern": "fresh", "run": i + 1, "time_s": t})

        print(f"\n=== {name} — Pattern B: reused store ===")
        times_reused = measure_reused(name, factory, N)
        results[(name, "reused")] = times_reused
        for i, t in enumerate(times_reused):
            rows.append({"backend": name, "pattern": "reused", "run": i + 1, "time_s": t})
        print()

    print_summary(results)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    csv_path = save_csv(rows, out_dir)
    print(f"\nResults saved to {csv_path}")
