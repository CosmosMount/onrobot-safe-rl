# Q_safe state-dependent recovery V4 technical-failure record

Status: **terminal tooling-invalid; no Stage-A scientific decision**  
Recorded outcome-blind at: `2026-08-09T14:59:13Z`  
Collection/generator commit: `8cb6a3d038c23361fc142e20a6a0d2ad42c9df7f`  
Protocol contract SHA-256: `101484a5df78b22941a8988f9936c7fb40b4569ed5c555273843484275dcc977`  
Protocol file SHA-256: `dc11b0267042448434076caf359dac7da039e9eec256cb8d04fa09d87d32505f`  
Cohort-lock SHA-256: `e1d6d697d410ac4820672cd49d89a2f280d221462ba7e8abf3522e363d686cff`

## What completed

All six preregistered source collectors completed once with exit code zero and
published their report-last artifacts. No source was topped up or rerun.

| Source seed | Groups | Proposals | Source steps | Trajectories | Attempt-marker SHA-256 |
|---:|---:|---:|---:|---:|---|
| 8401 | 64 | 1657 | 9570 | 303 | `086526bf6962d4598e108084b0ff9365ece617c6fd634b1cd74d52c02fa0897a` |
| 8402 | 64 | 1778 | 10326 | 332 | `58a1439ea4c7a8368ce3b98331406dfcdccbd41b6d5b18bf851809eb2dc87d2c` |
| 8411 | 64 | 1596 | 9097 | 284 | `8f62e510966902437a5a281a8c371af9d95b80216ac79d859fb9e7b41e7c3134` |
| 8412 | 64 | 1568 | 8991 | 288 | `96ee18773cab0d0e05874470182cba6bf7a9099a298582accb681dee337862dd` |
| 8421 | 64 | 1468 | 9036 | 351 | `288d54d514e1089b584f5acaf084a807522c805a9c5321181bb236742854be1f` |
| 8422 | 64 | 1170 | 7146 | 276 | `512a743b9b19f0e63ae87e927ccc2ebff4a87d2236ab2d6943926513e59a75d6` |

The admission merge then completed on the same clean commit. It accepted the
preregistered 384 groups from 9237 proposals, published its report last, and
reported `candidate_outcomes_opened=false` and `audit_opened=false`. Its
collection-readiness SHA-256 was
`b4bb13554d6b815f798f9f039a75c5f06bf67f067e5a2aa8ad4875cd10095201`.

## Terminal failure

The first and only V4 discovery-merge invocation exited nonzero before the
merge, data gate, staging, selection, or audit steps:

```text
ValueError: discovery leaf does not bind the V4 RNG/split
```

The process loaded and schema-validated the first discovery deployable and
privileged leaf before evaluating the failing manifest predicate. It did not
combine discovery shards, calculate any discovery risk or informativeness
statistic, select a candidate, create a selection lock, or open an audit shard.
The three no-clobber discovery aggregate/report destinations remained absent.

Static diagnosis found a deterministic producer/consumer contract bug. The
collector correctly wrote the exact V4 seed manifest with four fields:
`domain_hex`, `role_tags`, `algorithm`, and `stream_mapping`. The wrapper-local
merge helper expected only the first two fields and compared the dictionaries
for exact equality. The canonical downstream lock validator independently
expects the same complete four-field manifest written by the collector.
Consequently every valid V4 discovery leaf would be rejected; this was not a
seed-specific or data-integrity failure.

## Scientific disposition

This V4 iteration is permanently **tooling-invalid**. It provides no Stage-A
pass/fail result and does not authorize model training, Stage B, Objective 1,
or Phase 2. Seeds 8401, 8402, 8411, 8412, 8421, and 8422 remain consumed under
the preregistered no-top-up/no-rerun rule. Their artifacts must not be merged
into or reused as confirmatory data for a later iteration.

An in-place patch, runtime monkeypatch, manifest edit, or ordinary rerun is
forbidden: the V4 protocol deliberately requires merge, lock, and audit to run
from the exact clean collection commit, while any honest repair changes that
commit. `resume-denied-report` is inapplicable because no selection lock was
created.

The next confirmatory attempt must therefore:

1. share one canonical four-field RNG-manifest helper between writer and
   merger and add producer-to-merger integration and negative tests;
2. use a new protocol identity, RNG domain, artifact root, and six fresh source
   seeds;
3. keep G/K/R/H, policy ages, candidate definitions/order, statistical gates,
   bootstrap rules, and selection semantics unchanged from V4; and
4. complete fresh outcome-free preflights before any new collection.

Until that fresh attempt passes, Objective 1 remains unmet and Objective 2 is
blocked.
