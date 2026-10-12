# Optional routing-weighted Q256 scheduling

Set `AITER_Q256_WEIGHTED_SCHEDULE=1` before starting Python to enable the
routing-weighted scheduler for compact Qwen prefill with 8192 tokens, model
width 8192, intermediate width 256, 513 experts, and top-k 11. The default is
disabled. Other shapes, including 32768 tokens, use the existing producer.
Direct callers may also pass `weighted_schedule=True` to `fmoe_q256`.

The scheduler builds a 1024-entry int32 job map on the GPU for every call,
including graph replay. It estimates live routed fragments, orders jobs by
cost within each XCD, and alternates owner assignment across passes. The
producer and next-job prefetch use the same map. Contributions and reduction
arithmetic remain unchanged; the shared FlyDSL reducer handles the output.
The map consumes 4096 bytes and uses the reserved pointer slot at ABI offset
128. The C++ launcher checks its shape, dtype, contiguity, device, and geometry.

The weighted object has its own `_route_weighted` symbol and filename. SHA256:
`96e877979784bac87588f44f02ee66bbab052b1c7bf6cbb17bd5c33ca4800f25`.

## Validation

On MI355X/gfx950 with 256 CUs, two reversed fresh-process repetitions of the
random-routing full-call test produced the following means of process medians.
Each process median uses seven samples of 50 graph replays after warmup; map
construction is included.

| Tokens | Weighted opt-in, us | Fixed Q256, us | FlyDSL, us |
| ---: | ---: | ---: | ---: |
| 8192 | 1091.134 | 1090.821 | 1247.891 |
| 32768 | 3734.240 | 3723.559 | 4462.936 |

Weighted scheduling is near parity with fixed Q256 on this random fixture.
The 32768-token opt-in arm is a fallback control running the fixed producer,
so its difference is run variation. All twelve reference/API cases passed;
all eight paired complete outputs were exactly equal. Profiles confirm map
construction and the weighted symbol at 8192, and neither at 32768.

Separately, supplied captured-routing replay results report a 7.04% reduction
in the sum of per-capture full-call medians across eight Qwen routing captures
against fixed Q256. Those tests use synthetic activation/weight tensors and
16 samples of 100 replays in paired blocks. They are a different protocol from
the random-routing table and do not establish full-model serving latency.

Use a fresh `AITER_JIT_DIR` after installing the launcher and object. With the
Q256 selector CSV configured, the existing validation wrapper can be run with:

```sh
AITER_Q256_WEIGHTED_SCHEDULE=1 python3 op_tests/test_q256_aiter.py \
  --aiter-root "$PWD" --layout compact --compare-producer \
  --tokens 8192 --output /tmp/q256-weighted-8192.json
```

The wrapper requires `AITER_CONFIG_FMOE` and `AITER_JIT_DIR`. Run the same command
with the opt-in set to `0` for the fixed control. Full-model latency and memory
validation are still required before enabling the opt-in in production.


## Captured routing limits at 8192 tokens

A three-provider paired replay on MI355X PCI `0000:65:00.0` used the same
captured token/slot routes and gates with synthetic activation/expert tensors.
Each result below is the median of 15 paired percentages, with ABBA/BAAB order
and 50 full API calls per block. Exact complete outputs agree among weighted
Q256, fixed Q256 and FlyDSL; sampled reference NRMS is below 0.002627.

| Capture | Weighted vs fixed Q256, eager | Weighted vs fixed Q256, graph | Weighted vs FlyDSL, eager | Weighted vs FlyDSL, graph |
| --- | ---: | ---: | ---: | ---: |
| 029 | +4.315% | +3.622% | +6.677% | +7.312% |
| 046 | +1.535% | +1.527% | -1.152% | +0.211% |
| 049 | +2.593% | +3.706% | -1.137% | -0.950% |

Positive percentages mean weighted Q256 is faster. Graph capture 046 is near
parity with FlyDSL and has a broad paired range (-3.023% to +3.477%); capture 049
loses every sample in both modes. A preceding independent replay also lost all
15 samples on 046/049 against FlyDSL, at about 1% in graph mode. Scheduling helps
against fixed Q256 while the complete Q256 pipeline can still lose to FlyDSL.

Captured layouts 029/046/049 have 595/680/675 routed BM256 tiles and 723/808/803
logical jobs. The latter two need a fourth job for 40/35 persistent owners.
The existing routed GU selects at least M128, so tails with ≤64 live rows still
execute additional GU capacity. This association does not isolate padding from
expert distribution or weight locality.

The production weighted CO is unchanged. Experimental M32/M64 routed variants
are excluded from this PR. Trimming GU and down passed correctness but worsened
046/049 latency; trimming only GU retained about a 1% FlyDSL deficit on those
captures. These experiments do not establish padding as the sole cause. Full model validation is still required before default enablement.

Per-sample timing, artifact hashes and correctness evidence are recorded in
[q256_weighted_real_routes_20261012.json](benchmarks/q256_weighted_real_routes_20261012.json).
