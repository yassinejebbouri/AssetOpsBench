# Fix 3: Parallelize N×M Mapping Loop — Log

**Date:** 2026-04-20  
**Branch:** performance-profiling  
**File changed:** `src/servers/fmsr/main.py`

---

## The Root Bottleneck

Every FMSR scenario ultimately calls `get_failure_mode_sensor_mapping`, which must determine — for every (failure_mode, sensor) pair — whether that sensor can detect that failure. IBM's own docstring warned:

> "one LLM call is made per (failure_mode, sensor) pair sequentially... For a chiller with 7 failure modes and 20+ sensors the call will take **several minutes**."

For a Chiller with 5 failure modes and 10 sensors: **50 serial WatsonX calls**.  
At ~0.5–1s per call: **25–50 seconds just for the mapping step**, before any overhead.

The Fix 1 benchmark confirmed this — FMSR scenarios averaged **8.26s** with some hitting **12–14s**.  
The WatsonX 429 rate limit errors seen throughout Fix 2 are further evidence: sequential bursts were saturating the free-tier limit (2 req/s, 10 concurrent).

---

## What Changed

### `src/servers/fmsr/main.py`

**1. New imports:** `asyncio`, `time`

**2. New constant:**
```python
_CONCURRENCY = int(os.environ.get("FMSR_CONCURRENCY", "8"))
```
Caps concurrent WatsonX calls at 8 (just under the free-tier 10-concurrent limit). Tunable via env var.

**3. Exponential backoff in `_call_relevancy`:**

Before:
```python
for _ in range(_MAX_RETRIES):
    try:
        return _parse_relevancy(_llm.generate(prompt))
    except Exception as exc:
        last_exc = exc
raise last_exc
```

After:
```python
for attempt in range(_MAX_RETRIES):
    try:
        return _parse_relevancy(_llm.generate(prompt))
    except Exception as exc:
        last_exc = exc
        if attempt < _MAX_RETRIES - 1:
            time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s, 2s
raise last_exc
```

Previously, retries on 429 errors happened instantly — which just got another 429 immediately. The backoff gives WatsonX time to recover between retries.

**4. `get_failure_mode_sensor_mapping` converted from sync → async:**

Before (sequential):
```python
def get_failure_mode_sensor_mapping(...):
    for s in sensors:
        for fm in failure_modes:
            gen = _call_relevancy(asset_name, fm, s)   # blocking, one at a time
            ...
```

After (parallel):
```python
async def get_failure_mode_sensor_mapping(...):
    semaphore = asyncio.Semaphore(_CONCURRENCY)         # max 8 concurrent

    async def _one_pair(s, fm):
        async with semaphore:
            gen = await asyncio.to_thread(_call_relevancy, asset_name, fm, s)
        return RelevancyEntry(...)

    entries = await asyncio.gather(*[_one_pair(s, fm) for s in sensors for fm in failure_modes])
```

`asyncio.to_thread` runs the blocking `_llm.generate()` call in a thread pool so it doesn't block the event loop. `asyncio.gather` fires all N×M coroutines concurrently, but the semaphore ensures at most 8 are in-flight at any moment.

---

## Expected Impact

| Scenario | Before (serial) | After (parallel, 8 concurrent) |
|----------|----------------|-------------------------------|
| 5 FM × 10 sensors = 50 pairs | ~50s (1s/call) | ~7s (50/8 batches × ~1s) |
| 3 FM × 5 sensors = 15 pairs | ~15s | ~2s (2 batches) |
| Latency reduction | — | ~5–7× speedup |
| 429 rate limit errors | frequent | reduced (backoff + semaphore) |

**Predicted benchmark improvement:** FMSR average latency drops from ~8–14s to ~2–4s.  
Accuracy should be unchanged (same LLM calls, same logic, just concurrent).

---

## How to Tune

```bash
# Increase concurrency if your WatsonX plan allows more concurrent requests
FMSR_CONCURRENCY=10 uv run profiling-benchmark --orchestrators MetaAgent --domains fmsr

# Lower it if you're still hitting rate limits on free tier
FMSR_CONCURRENCY=4 uv run profiling-benchmark --orchestrators MetaAgent --domains fmsr
```

---

## Why Not Just Cache?

Caching (Fix 5) would help for repeated queries about the same (asset, failure_mode, sensor) triplet. But in the benchmark, each scenario has a unique combination, so cache hit rate would be near 0%. Fix 3 (parallelization) helps every single invocation regardless of caching.
