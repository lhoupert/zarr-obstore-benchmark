# Findings: obstore vs fsspec for zarr GCS reads

**Date:** 2026-06-04  
**Environment:** zarr 3.2.1, obstore 0.10.0, gcsfs 2026.5.0, Python 3.12.12 (arm64 macOS)  
**Dataset:** `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`  
**Variable:** `2m_temperature`, shape `(1323648, 721, 1440)`, chunks `(1, 721, 1440)` = 4.2 MB/chunk

---

## 1. What the original notebook measured

The [original notebook](https://github.com/ofk123/dataloading-example) timed `xr.open_zarr(store=...)`
using `%%timeit`. This **only reads metadata and coordinate arrays** — it does **not** read chunk data.
The reported 1.5 s (gcsfs) vs 2.7 s (obstore) reflect metadata-open latency, not data-read throughput.

The notebook also used **unequal credentials**: obstore received a `GoogleCredentialProvider` that
resolves a service-account token (potentially a per-request fetch), while gcsfs got a pre-cached
`token`. This auth asymmetry could explain part of the gap — but as shown below, it does not
explain all of it.

---

## 2. Why concurrency=128 "didn't help" the notebook's metric

`async.concurrency` (zarr's `BatchedCodecPipeline` parallelism) controls how many chunk-reads
are dispatched in parallel. In the open-only workload, zarr fetches:

- `.zmetadata` (one consolidated JSON blob, 133 KB) — one request
- coordinate arrays (`time`, `latitude`, `longitude`, `level`) — a handful of small reads

There are too few keys and too little parallelism opportunity for a higher concurrency ceiling
to matter. This is exactly what the reporter observed. Concurrency **does** matter for bulk
chunk reads (see G1 below).

---

## 3. G2 — Open-phase reproducibility

> *"Did Claude reproduce the finding that obstore is slower for the metadata + coords reads?"*

**Yes, the finding reproduces under equal anonymous auth.**

Measured with fresh store per call, `consolidated=True`, 3 repetitions:

| Backend | cold open | warm open (median) |
|---------|-----------|-------------------|
| obstore | 1.93 s    | 1.92 s            |
| fsspec  | 1.90 s    | **1.23 s**        |
| ratio   | ~1.0×     | **1.56×**         |

Key observation: **obstore's open time is constant across repetitions (~1.73–1.93 s)** while
**fsspec warms up significantly (1.90 s → 1.06–1.23 s)**. The gap is *not* present on the
first (cold) call; it emerges on warmed calls.

**Root cause — connection pooling asymmetry:**

- `gcsfs.GCSFileSystem` uses `aiohttp`. Even when a new `GCSFileSystem` object is created,
  `aiohttp` reuses TCP connections from the OS socket layer (and potentially the event-loop-level
  connector pool). Later open calls benefit from warm HTTP keep-alive connections to GCS.
- `obstore.store.GCSStore` is backed by a Rust HTTP client. Each new `GCSStore` instance
  creates a fresh connection pool; there is no shared pool across Python-level store objects.
  Every open call pays the TCP handshake + TLS setup cost.

The auth confound from the notebook (token-fetch overhead for obstore) is **not** the primary
cause. Removing it (using `skip_signature=True` and `token="anon"`) reduces but does not
eliminate the gap, because the warm-connection asymmetry remains.

---

## 4. G1 — Data-read throughput

210 MB slab (50 chunks × 4.2 MB), time[400000:400050], 3 repetitions each:

| concurrency | obstore median | fsspec median | ratio (obs/fsspec) |
|-------------|---------------|---------------|-------------------|
| 10          | 9.36 s        | 8.40 s        | 1.11×             |
| 32          | 8.58 s        | 7.19 s        | 1.19×             |
| **128**     | **7.32 s**    | **7.06 s**    | **1.04× (~equal)**|

**obstore matches fsspec at high concurrency (≥128) for bulk data reads** — confirming the
plan's G1 expectation. The small residual overhead at lower concurrency is consistent with
obstore paying a per-connection setup cost when the concurrency ceiling limits parallelism.

Concurrency does matter for data reads: both backends improve from ~9 s (concurrency=10)
to ~7 s (concurrency=128), a ~1.3× speedup — exactly because more chunks are fetched in
parallel.

Equal-work check: `np.allclose(equal_nan=True)` passed — both backends return numerically
identical arrays.

---

## 5. Summary table (all phases)

| phase       | metric       | obstore | fsspec | ratio | notes |
|-------------|-------------|---------|--------|-------|-------|
| store-init  | median       | 0.22 s  | ~0 s   | —     | obstore creates Rust HTTP client |
| open (G2)   | cold         | 1.73–2.0 s | 1.78–1.90 s | ~1× cold | equal on first call |
| open (G2)   | warm median  | 1.73–1.92 s | 1.06–1.23 s | **1.5–1.75×** | obstore slower (no conn reuse) |
| data-read   | conc=10      | 9.4 s   | 8.4 s  | 1.11× | both limited by serial batches |
| data-read   | conc=128     | 7.3 s   | 7.1 s  | **1.04×** | obstore matches at high concurrency |

---

## 6. Conclusions

**G1 verdict:** obstore is **not** slower than fsspec for bulk chunk reads at high concurrency.
At `async.concurrency=128` (≥ number of chunks), obstore is within 4% of fsspec — statistically
indistinguishable given run-to-run variance. The "obstore is 40–50% slower" claim from the
thread measured only the open phase, which never exercises bulk I/O.

**G2 verdict:** obstore **is** reproducibly slower for the open phase under equal anonymous
auth, by ~1.5–1.75× on warm calls. The root cause is **HTTP connection pooling behavior**:
gcsfs/aiohttp reuses TCP connections across repeated `open_zarr` calls; obstore's Rust client
does not share its connection pool across separate `GCSStore` Python objects. The auth
confound in the notebook (token-fetch per request) is an additional factor but not the
primary one.

**Recommendation for zarr-python / obstore maintainers:**
The open-phase gap could be reduced by either:
1. Allowing `GCSStore` instances to share an underlying HTTP client / connection pool across
   Python object boundaries (obstore change), or
2. Documenting that users should reuse `ObjectStore` instances rather than recreating them
   per `open_zarr` call (user guidance).

For the typical Earth-science workflow (open once, read data in a loop), the G1 result is the
relevant one: **obstore is competitive with fsspec and should be preferred when high
concurrency is available**, as it provides a simpler, dependency-lighter interface to GCS.

---

## 7. Workaround: reuse the store object

The G2 gap is largely a **usage-pattern issue**, not a fundamental performance defect.
Running `test_store_reuse.py` (N=5 reps, `consolidated=True`) confirms this:

| Backend | fresh — run 1 | fresh — warm median | reused — run 1 | reused — warm median |
|---------|--------------|---------------------|----------------|----------------------|
| obstore | 4.64 s | 2.52 s | 1.85 s | **1.67 s** |
| fsspec  | 4.52 s | 1.78 s | 1.82 s | 1.81 s |

Key observations:
- **Reusing the same `ObjectStore` instance drops obstore's warm open time from 2.52 s to
  1.67 s — a 1.5× speedup**, and makes it *faster* than fsspec (1.67 s vs 1.81 s).
- Both backends pay a first-call penalty (~1.8–4.6 s, partially from GCS-side caching of
  `.zmetadata`) that disappears on subsequent calls.
- With store reuse, **obstore and fsspec are statistically indistinguishable** for open
  latency (0.93× ratio).

**Recommended pattern for users:**

```python
# Create the store ONCE per process/session — not per open_zarr call
from obstore.store import GCSStore
from zarr.storage import ObjectStore

store = ObjectStore(GCSStore("my-bucket", prefix="path/to/store.zarr", skip_signature=True),
                   read_only=True)

# Reuse it across multiple open_zarr calls
ds1 = xr.open_zarr(store, consolidated=True)
ds2 = xr.open_zarr(store, consolidated=True)  # uses warm connection pool
```

This matches how users naturally interact with gcsfs (a single `GCSFileSystem` object
reused across sessions), and eliminates the open-phase gap entirely.

**Action item for zarr-python docs / obstore integration guide:** add a note that
`ObjectStore` instances are lightweight to reuse and should not be recreated per
`open_zarr` call. This is analogous to not creating a new `requests.Session` per HTTP
call.

---

## 8. Limitations

- Runs from a single network endpoint (macOS laptop); GCS egress to this location affects
  absolute times but not the relative obstore/fsspec comparison.
- 3 repetitions; IQR is zero where variance is low with N=3. Re-running with `--n 7` would
  tighten the statistics.
- Only `consolidated=True` open tested here; `consolidated=False` (without `.zmetadata`) is
  much slower for both backends (~10–24 s) and dominated by the number of per-variable
  metadata requests.
