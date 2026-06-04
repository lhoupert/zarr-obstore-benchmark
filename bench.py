"""
bench.py — zarr obstore vs fsspec/gcsfs benchmark on GCS ERA5 data.

Separates the three phases that the original notebook conflated:
  1. store-init  — time to construct the Store object
  2. open (G2)   — xr.open_zarr(), metadata + coord reads only
  3. data-read (G1) — .isel(time=SLAB).load(), ~210 MB of chunk I/O

Usage:
    python bench.py                              # both adapters, default settings
    python bench.py --adapter obstore
    python bench.py --adapter fsspec
    python bench.py --concurrency 10 32 128 256
    python bench.py --n 5 --data-chunks 50

Results are written to results/<timestamp>_<adapter>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import zarr
import xarray as xr
import gcsfs
from obstore.store import GCSStore
from zarr.storage import FsspecStore, ObjectStore

# ── Dataset ───────────────────────────────────────────────────────────────────
BUCKET = "gcp-public-data-arco-era5"
PATH = "ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DATA_VAR = "2m_temperature"           # 3-D (time, lat, lon), chunk=4.2 MB
# time[400000] is in a valid (non-NaN) region of the dataset
DATA_TIME_START = 400_000


def make_obstore() -> ObjectStore:
    return ObjectStore(GCSStore(BUCKET, prefix=PATH, skip_signature=True), read_only=True)


def make_fsspec() -> FsspecStore:
    fs = gcsfs.GCSFileSystem(token="anon")
    return FsspecStore(fs, path=f"{BUCKET}/{PATH}")


STORE_FACTORIES = {
    "obstore": make_obstore,
    "fsspec": make_fsspec,
}


# ── Timed phases ──────────────────────────────────────────────────────────────

def time_store_init(factory) -> float:
    t0 = time.perf_counter()
    s = factory()
    return time.perf_counter() - t0, s


def time_open(store) -> tuple[float, xr.Dataset]:
    t0 = time.perf_counter()
    ds = xr.open_zarr(store, consolidated=True)
    return time.perf_counter() - t0, ds


def time_data_read(store, n_chunks: int) -> tuple[float, np.ndarray]:
    _, ds = time_open(store)
    t0 = time.perf_counter()
    arr = ds[DATA_VAR].isel(time=slice(DATA_TIME_START, DATA_TIME_START + n_chunks)).load()
    elapsed = time.perf_counter() - t0
    return elapsed, arr.values


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_phase(name: str, factory, concurrency: int, n_reps: int, n_chunks: int) -> dict:
    """Run all three phases for one adapter at one concurrency level."""
    zarr.config.set({"async.concurrency": concurrency})

    init_times, open_times, read_times = [], [], []
    ref_arr = None

    for rep in range(n_reps):
        # store-init
        t_init, store = time_store_init(factory)
        init_times.append(t_init)

        # open (G2)
        t_open, _ = time_open(factory())
        open_times.append(t_open)

        # data-read (G1) — fresh store per rep
        t_read, arr = time_data_read(factory(), n_chunks)
        read_times.append(t_read)

        if ref_arr is None:
            ref_arr = arr
        else:
            ok = np.allclose(arr, ref_arr, equal_nan=True)
            if not ok:
                print(f"  WARNING: rep {rep} data mismatch for {name} concurrency={concurrency}")

        mb = n_chunks * 4.2
        print(
            f"  {name:8s} concurrency={concurrency:4d} rep={rep+1}/{n_reps}  "
            f"init={t_init:.3f}s  open={t_open:.2f}s  "
            f"read={t_read:.2f}s ({mb:.0f} MB)"
        )

    def stats(vals):
        med = statistics.median(vals)
        iqr = statistics.quantiles(vals, n=4)[2] - statistics.quantiles(vals, n=4)[0] if len(vals) >= 4 else 0.0
        return {"cold": vals[0], "median": med, "iqr": iqr, "all": vals}

    return {
        "adapter": name,
        "concurrency": concurrency,
        "n_reps": n_reps,
        "n_chunks": n_chunks,
        "data_mb": n_chunks * 4.2,
        "init": stats(init_times),
        "open": stats(open_times),
        "read": stats(read_times),
    }


def equal_work_check(n_chunks: int) -> bool:
    """Confirm both backends return numerically identical arrays."""
    print("Equal-work check (np.allclose equal_nan=True) ...")
    zarr.config.set({"async.concurrency": 32})
    _, arr_obs = time_data_read(make_obstore(), n_chunks)
    _, arr_fsspec = time_data_read(make_fsspec(), n_chunks)
    ok = np.allclose(arr_obs, arr_fsspec, equal_nan=True)
    print(f"  Arrays {'MATCH' if ok else 'DO NOT MATCH'} (shapes: {arr_obs.shape}, {arr_fsspec.shape})")
    return ok


# ── Output ────────────────────────────────────────────────────────────────────

def save_results(results: list[dict], out_dir: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    adapters = "_".join(sorted({r["adapter"] for r in results}))
    csv_path = out_dir / f"{ts}_{adapters}.csv"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["adapter", "concurrency", "n_chunks", "data_mb",
             "init_cold", "init_median",
             "open_cold", "open_median", "open_iqr",
             "read_cold", "read_median", "read_iqr"]
        )
        for r in results:
            w.writerow([
                r["adapter"], r["concurrency"], r["n_chunks"], r["data_mb"],
                r["init"]["cold"], r["init"]["median"],
                r["open"]["cold"], r["open"]["median"], r["open"]["iqr"],
                r["read"]["cold"], r["read"]["median"], r["read"]["iqr"],
            ])

    json_path = csv_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return csv_path


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print(f"{'ADAPTER':8s} {'CONC':6s} {'INIT':8s} {'OPEN_cold':10s} {'OPEN_med':10s} {'READ_cold':10s} {'READ_med':10s}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['adapter']:8s} {r['concurrency']:6d} "
            f"{r['init']['cold']:.3f}s  "
            f"{r['open']['cold']:7.2f}s    {r['open']['median']:7.2f}s    "
            f"{r['read']['cold']:7.2f}s    {r['read']['median']:7.2f}s"
        )

    # G1 ratio at each concurrency
    print("\nG1 data-read ratio (obstore / fsspec), median:")
    conc_levels = sorted({r["concurrency"] for r in results})
    for c in conc_levels:
        obs = next((r for r in results if r["adapter"] == "obstore" and r["concurrency"] == c), None)
        fsspec = next((r for r in results if r["adapter"] == "fsspec" and r["concurrency"] == c), None)
        if obs and fsspec:
            ratio = obs["read"]["median"] / fsspec["read"]["median"]
            flag = "<-- obstore FASTER" if ratio < 0.95 else ("--> obstore slower" if ratio > 1.05 else "  ~equal")
            print(f"  concurrency={c:4d}: {ratio:.2f}x  {flag}")

    print("\nG2 open-phase ratio (obstore / fsspec), median:")
    for c in conc_levels:
        obs = next((r for r in results if r["adapter"] == "obstore" and r["concurrency"] == c), None)
        fsspec = next((r for r in results if r["adapter"] == "fsspec" and r["concurrency"] == c), None)
        if obs and fsspec:
            ratio = obs["open"]["median"] / fsspec["open"]["median"]
            flag = "<-- obstore FASTER" if ratio < 0.95 else ("--> obstore slower" if ratio > 1.05 else "  ~equal")
            print(f"  concurrency={c:4d}: {ratio:.2f}x  {flag}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="obstore vs fsspec zarr benchmark")
    parser.add_argument("--adapter", choices=["obstore", "fsspec", "all"], default="all")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[10, 32, 128])
    parser.add_argument("--n", type=int, default=3, help="repetitions per phase")
    parser.add_argument("--data-chunks", type=int, default=50,
                        help="number of chunks to read in G1 (default 50 = ~210 MB)")
    parser.add_argument("--skip-equal-work", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    import zarr as _zarr
    import obstore as _obs
    print(f"zarr {_zarr.__version__}  obstore {_obs.__version__}  gcsfs {gcsfs.__version__}")
    print(f"Dataset: gs://{BUCKET}/{PATH}  var={DATA_VAR}  time[{DATA_TIME_START}:{DATA_TIME_START+args.data_chunks}]")

    if not args.skip_equal_work:
        ok = equal_work_check(min(5, args.data_chunks))
        if not ok:
            print("FATAL: equal-work check failed — backends return different data. Aborting.")
            return

    adapters = ["obstore", "fsspec"] if args.adapter == "all" else [args.adapter]
    results = []

    for concurrency in args.concurrency:
        print(f"\n--- concurrency={concurrency} ---")
        for name in adapters:
            r = run_phase(name, STORE_FACTORIES[name], concurrency, args.n, args.data_chunks)
            results.append(r)

    print_summary(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_results(results, args.out_dir)
    print(f"\nResults saved to {csv_path} (+ .json)")


if __name__ == "__main__":
    main()
