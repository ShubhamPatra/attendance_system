# Benchmark Results

Generated from `scripts/benchmark_pipeline.py --frames 20` on April 15, 2026.

## Pipeline Summary

- Benchmark input: synthetic 640x480 frames
- Frame count: 20
- Profiling mode: optimized recognition pipeline with synthetic detector/tracker/embedder
- Memory tracking: `tracemalloc`

## Latency Table

| Stage | Count | Total ms | Avg ms | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|---:|---:|
| align | 20 | 0.08 | 0.00 | 0.00 | 0.00 | 0.02 |
| detect | 7 | 0.18 | 0.03 | 0.01 | 0.12 | 0.12 |
| embed | 20 | 0.05 | 0.00 | 0.00 | 0.01 | 0.01 |
| track | 13 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| track_init | 7 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

## ONNX Session Load Comparison

| Mode | Time (ms) |
|---|---:|
| Optimized | 14.18 |
| Baseline | 5.56 |

## Scaling Benchmark (FPS)

| Mode | Scaled | Unscaled |
|---|---:|---:|
| FAST | 6263.39 | 23744.06 |
| BALANCED | 5189.27 | 20897.49 |
| ACCURATE | 10626.99 | 11313.56 |

## Adaptive Interval Benchmark (FPS)

| Mode | FPS |
|---|---:|
| Adaptive detection interval | 3301.36 |
| Fixed detection interval | 3858.65 |

## Memory Usage

- Current: 19,262 bytes
- Peak: 578,300 bytes

## Notes

- The benchmark uses synthetic frames and mocked recognition stages, so the numbers are suitable for regression tracking rather than absolute production latency.
- Scaled vs unscaled FPS can appear inverted in this synthetic setup because detector/embedding costs are stubbed and `cv2.resize` overhead dominates.
- Adaptive interval vs fixed interval can also invert in this synthetic setup because the mocked detector is cheap and adaptive control logic overhead becomes more visible.
- GPU benchmarking was not executed in this environment.