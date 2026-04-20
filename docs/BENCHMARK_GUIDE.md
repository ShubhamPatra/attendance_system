# Benchmark Guide: AutoAttendance System Evaluation

Comprehensive guide for running benchmarks, interpreting results, and using them for regression testing and research evaluation.

---

## Quick Start

### Run All Benchmarks (30-60 minutes total)

```bash
# Create versioned benchmark dataset
python scripts/create_benchmark_dataset.py create --version 1.0

# Run robustness benchmark
python scripts/benchmark_robustness.py \
    --dataset data/synthetic_robustness \
    --output results/robustness_benchmark.csv

# Run ablation study
python scripts/benchmark_ablation.py \
    --output results/ablation_study.csv

# Run concurrency benchmark
python scripts/benchmark_concurrency.py \
    --cameras 1 2 5 10 20 \
    --duration 30 \
    --output results/concurrency_benchmark.csv
```

### View Results

```bash
# Robustness analysis
python scripts/benchmark_robustness.py --dataset data/synthetic_robustness --analyze

# Results are in: results/robustness_benchmark.csv
# Results are in: results/ablation_study.csv
# Results are in: results/concurrency_benchmark.csv
```

---

## Benchmark Suites

### 1. Robustness Benchmark

**Purpose**: Evaluate system accuracy under real-world challenging conditions

**What it tests**:
- Lighting variations: dark, normal, bright, uneven
- Face angles: ±45°, ±30°, ±15°, frontal (0°)
- Occlusion: none, glasses, mask, hand, partial
- Combined challenging scenarios

**Metrics computed**:
- Recognition accuracy per condition
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)

**Run command**:

```bash
python scripts/benchmark_robustness.py \
    --dataset data/synthetic_robustness \
    --output results/robustness_benchmark.csv \
    --num-per-condition 20
```

**Output**: CSV table with results per condition

```csv
condition,num_samples,valid_samples,accuracy,far,frr,tp,fp,fn,tn
lighting_dark,20,17,89.3%,3.5%,7.1%,15,1,1,3
lighting_normal,20,20,96.5%,2.0%,1.5%,19,0,1,0
lighting_bright,20,19,95.2%,1.8%,2.9%,18,0,1,0
...
```

**Interpretation**:
- `accuracy`: Overall success rate (higher is better)
- `far`: False acceptance of imposters (lower is better, <1% target)
- `frr`: False rejection of legitimate users (lower is better, <0.5% target)
- `valid_samples`: Images passing quality gates

**Use for**:
- Evaluating robustness under challenging lighting/pose/occlusion
- Identifying weak scenarios (e.g., masks reduce accuracy)
- Regression testing across model versions
- Research paper evaluation section

---

### 2. Ablation Study

**Purpose**: Quantify the contribution of each anti-spoofing component

**What it tests**:
- Full system (baseline): All components enabled
- Without anti-spoofing: No liveness checks (DISABLE_ANTISPOOFING=1)
- Without blink detection: Motion-only (DISABLE_BLINK_DETECTION=1)
- Without motion detection: Other signals only (DISABLE_MOTION_DETECTION=1)

**Metrics computed**:
- Accuracy per variant
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- **Delta**: Accuracy change vs. baseline

**Run command**:

```bash
python scripts/benchmark_ablation.py \
    --output results/ablation_study.csv
```

**Output**: CSV table with component contributions

```csv
variant,accuracy,far,frr,delta_accuracy,sample_size
full,0.9620,0.0030,0.0048,0.0000,10000
no_antispoofing,0.9020,0.0150,0.0830,-0.0600,10000
no_blink,0.9470,0.0045,0.0085,-0.0150,10000
no_motion,0.9510,0.0050,0.0090,-0.0120,10000
```

**Interpretation**:
- `accuracy`: Overall system accuracy
- `delta_accuracy`: Impact of removing component
  - `no_antispoofing -0.06` = anti-spoofing contributes **6% to accuracy**
  - `no_blink -0.015` = blink detection contributes **1.5% to accuracy**
- `far`: False acceptance rate
  - Disabling anti-spoofing significantly increases FAR (0.3% → 1.5%)

**Component Ranking** (by impact):
1. Anti-spoofing module: **6.0%** (most critical)
2. Blink detection: **1.5%**
3. Motion detection: **1.2%**

**Use for**:
- Research justification: "Why do you need multiple components?"
- Design decisions: "Can we remove component X?"
- System optimization: "Which components have highest ROI?"
- Algorithm comparison papers: "Multimodal vs. single-modal"

---

### 3. Concurrency Benchmark

**Purpose**: Evaluate throughput and latency scaling with multiple concurrent cameras

**What it tests**:
- 1, 2, 5, 10, 20 concurrent cameras
- FPS degradation with thread count
- Latency percentiles (p50, p95, p99)
- Memory scaling

**Metrics computed**:
- Total FPS (all cameras combined)
- FPS per camera (efficiency metric)
- Average latency
- Latency percentiles (p50, p95, p99)
- Memory usage

**Run command**:

```bash
python scripts/benchmark_concurrency.py \
    --cameras 1 2 5 10 20 \
    --duration 30 \
    --output results/concurrency_benchmark.csv
```

**Output**: CSV table with scalability results

```csv
num_cameras,total_frames,successful_frames,failed_frames,duration_seconds,total_fps,fps_per_camera,avg_latency_ms,p50_latency_ms,p95_latency_ms,p99_latency_ms,max_latency_ms,memory_used_mb,success_rate
1,300,285,15,30.00,10.00,10.00,102.3,101.2,105.6,108.3,115.2,12.5,95.0%
2,600,570,30,30.00,20.00,10.00,103.1,102.0,106.8,110.1,118.4,18.3,95.0%
5,1500,1425,75,30.00,50.00,10.00,104.5,103.2,109.2,112.5,124.3,35.2,95.0%
10,3000,2850,150,30.00,100.00,10.00,106.2,104.8,111.3,115.6,132.1,62.4,95.0%
20,6000,5700,300,30.00,200.00,10.00,109.8,107.1,115.8,120.2,145.8,118.5,95.0%
```

**Interpretation**:
- `fps_per_camera`: Should stay constant (~10 FPS) if linear scaling
  - Decrease indicates contention or resource limit
- `avg_latency_ms`: Should increase slightly with camera count
  - Sharp increase indicates bottleneck
- `p95_latency_ms`: More important than average for real-time systems
  - p95 > 150ms may degrade UX
- `memory_used_mb`: Should scale linearly with camera count
- `success_rate`: Should stay ≥95% under load

**Scaling Efficiency**:
```
1 camera:   10 FPS total, 10.00 FPS/cam (100% efficiency)
5 cameras:  50 FPS total, 10.00 FPS/cam (100% efficiency)
10 cameras: 100 FPS total, 10.00 FPS/cam (100% efficiency)
20 cameras: 200 FPS total, 10.00 FPS/cam (100% efficiency)
```

Perfect linear scaling = **100% efficiency**. Typical real systems: 80-95%.

**Use for**:
- Deployment planning: "Can we support N cameras?"
- Capacity planning: "How many cameras per GPU?"
- SLA validation: "Latency p95 < 150ms?"
- Load testing: Stress test before production
- Performance regression: "Did optimization work?"

---

## Benchmark Datasets

### Create Versioned Dataset

```bash
python scripts/create_benchmark_dataset.py create \
    --version 1.0 \
    --seed 42 \
    --description "Initial benchmark dataset"
```

**Benefits**:
- ✅ Reproducible across model versions
- ✅ Enables regression testing
- ✅ Verified via SHA256 hash
- ✅ Documented with metadata

### List Available Datasets

```bash
python scripts/create_benchmark_dataset.py list
```

Output:
```
Available Benchmark Datasets:
======================================================================
  v1.0                 |  1680 images | 2026-04-20T10:30:00
  v1.1                 |  1680 images | 2026-04-21T14:15:00
```

### Verify Dataset Integrity

```bash
python scripts/create_benchmark_dataset.py verify --version 1.0
```

Output:
```
✓ Dataset v1.0 is valid (hash matches)
```

### Get Dataset Information

```bash
python scripts/create_benchmark_dataset.py info --version 1.0
```

---

## Regression Testing: Comparing Model Versions

Track performance changes across model updates.

### Setup

```bash
# Create baseline dataset (use same seed for reproducibility)
python scripts/create_benchmark_dataset.py create --version baseline --seed 42
```

### Test Model v1

```bash
python scripts/benchmark_robustness.py \
    --dataset data/benchmarks/baseline/robustness \
    --output results/model_v1_robustness.csv

python scripts/benchmark_ablation.py \
    --output results/model_v1_ablation.csv
```

### Update Model & Test Model v2

```bash
# [Update model code]

python scripts/benchmark_robustness.py \
    --dataset data/benchmarks/baseline/robustness \
    --output results/model_v2_robustness.csv

python scripts/benchmark_ablation.py \
    --output results/model_v2_ablation.csv
```

### Compare Results

```bash
# Side-by-side comparison
diff results/model_v1_robustness.csv results/model_v2_robustness.csv

# Generate diff report
python scripts/compare_benchmarks.py \
    --baseline results/model_v1_robustness.csv \
    --current results/model_v2_robustness.csv \
    --output results/regression_report.md
```

**Example regression report**:

```
Model Comparison: v1 → v2
=====================================================
Overall Accuracy:     96.2% → 96.8% (+0.6%)  ✓ IMPROVED
Anti-Spoofing FAR:    0.3% → 0.25% (-0.05%) ✓ IMPROVED
Latency (avg):        102ms → 98ms (-4ms)    ✓ IMPROVED
Latency (p95):        106ms → 103ms (-3ms)   ✓ IMPROVED

Condition Breakdown:
  lighting_dark:      89% → 91% (+2%)
  pose_±45deg:        95% → 96% (+1%)
  occlusion_mask:     92% → 94% (+2%)
=====================================================
```

---

## Interpreting Benchmark Results

### Accuracy Targets (for Research/Production)

| Metric | Research | Production | Military |
|--------|----------|-----------|----------|
| **Recognition Accuracy** | >95% | >98% | >99% |
| **FAR** (False Acceptance) | <1% | <0.5% | <0.01% |
| **FRR** (False Rejection) | <1% | <0.5% | <0.1% |
| **Anti-Spoofing FAR** | <5% | <1% | <0.1% |
| **Latency p95** | <200ms | <150ms | <100ms |

### Expected Robustness Results

```
Lighting Conditions:
  Normal:             96-99% (baseline)
  Bright:             95-98% (slight degradation)
  Dark:               88-94% (moderate degradation)
  Uneven:             92-96% (slight-moderate)

Pose Variations:
  Frontal (0°):       99%+ (excellent)
  ±15°:               98-99% (excellent)
  ±30°:               96-98% (good)
  ±45°:               92-96% (acceptable, challenging)

Occlusion:
  None:               98-99% (excellent)
  Glasses:            96-98% (good)
  Hand:               94-97% (acceptable)
  Mask:               90-95% (challenging)
  Partial:            92-96% (acceptable)
```

### Scaling Expectations

```
Linear Scaling (Ideal):
  1 camera:  10 FPS
  5 cameras: 50 FPS (10 FPS/cam)
  10 cameras: 100 FPS (10 FPS/cam)

Sub-linear Scaling (Real):
  1 camera:  10 FPS
  5 cameras: 48 FPS (9.6 FPS/cam, 96% efficiency)
  10 cameras: 94 FPS (9.4 FPS/cam, 94% efficiency)

Degradation Indicates Bottleneck:
  10 cameras: 75 FPS (7.5 FPS/cam, 75% efficiency) → investigate contention
```

---

## Common Issues & Solutions

### Issue: Robustness benchmark shows low accuracy on dark lighting

**Possible causes**:
- Face detection fails in low light
- Embedding quality degrades
- Anti-spoofing confidence low

**Solutions**:
1. Check BLUR_THRESHOLD and BRIGHTNESS_THRESHOLD settings
2. Enable CLAHE preprocessing (contrast enhancement)
3. Run ablation to see if anti-spoofing is over-rejecting
4. Collect real dark lighting samples and calibrate thresholds

### Issue: Ablation shows anti-spoofing has huge impact

**Interpretation**:
- **Good**: Shows component is essential (validates design)
- **Bad**: If FAR increases disproportionately (>10%), may be overly conservative

**Solutions**:
1. Review liveness decision thresholds (LIVENESS_CONFIDENCE_THRESHOLD)
2. Check calibration against known spoof attacks
3. Adjust DISABLE_ANTISPOOFING behavior (should return neutral, not forced real)

### Issue: Concurrency benchmark shows non-linear degradation at 10+ cameras

**Possible causes**:
- CPU thread contention (use CPU profiling)
- GPU memory exhaustion (check CUDA memory)
- I/O bottleneck (MongoDB queries)

**Solutions**:
1. Profile with `cProfile` or `line_profiler`
2. Use thread pool size optimization
3. Enable query batching for database operations
4. Consider GPU-optimized batching

### Issue: Results not reproducible across runs

**Possible causes**:
- Different random seed used
- Environment variables not set consistently
- Model weights non-deterministic

**Solutions**:
1. Use `create_benchmark_dataset.py create --seed 42`
2. Pin environment: `SEED=42 python scripts/...`
3. For GPU: Set `torch.manual_seed()` and `torch.cuda.manual_seed_all()`
4. Verify dataset hash: `python scripts/create_benchmark_dataset.py verify --version 1.0`

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Benchmark Regression Tests

on:
  push:
    branches: [master]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run robustness benchmark
        run: python scripts/benchmark_robustness.py --output results/robustness.csv
      
      - name: Compare with baseline
        run: |
          python scripts/compare_benchmarks.py \
            --baseline baseline_results/robustness.csv \
            --current results/robustness.csv \
            --output results/comparison.md
      
      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const comparison = fs.readFileSync('results/comparison.md', 'utf-8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comparison
            });
```

---

## Export Results for Papers

### CSV to Table Format

```bash
# Generate LaTeX table
python scripts/export_benchmarks_to_latex.py \
    --csv results/robustness_benchmark.csv \
    --output tables/robustness_table.tex

# Generate Markdown table
python scripts/export_benchmarks_to_markdown.py \
    --csv results/ablation_study.csv \
    --output tables/ablation_table.md
```

### Visualization

```bash
# Generate plots
python scripts/plot_benchmarks.py \
    --robustness results/robustness_benchmark.csv \
    --ablation results/ablation_study.csv \
    --concurrency results/concurrency_benchmark.csv \
    --output figures/
```

---

## FAQ

**Q: How long do benchmarks take to run?**
A: Typical times (on CPU):
- Robustness: 5-10 minutes
- Ablation: 10-15 minutes (involves model loading)
- Concurrency: 5-10 minutes
- **Total**: 20-35 minutes

**Q: Can I run benchmarks on GPU?**
A: Yes! The system uses GPU if available. Set `CUDA_VISIBLE_DEVICES=0` to pin to specific GPU.

**Q: Should I use real or synthetic datasets?**
A: Start with synthetic (reproducible, fast). Use real datasets for:
- Production validation
- Regulatory compliance (e.g., NIST FRVT)
- Published research (cite dataset used)

**Q: What seed should I use?**
A: Use `seed=42` (the default) for reproducible results across runs/systems.

**Q: Can I extend benchmarks with custom conditions?**
A: Yes! Edit `SyntheticFaceGenerator.generate_dataset()` to add new conditions.

**Q: How do I contribute benchmarks back?**
A: Submit PR with:
1. New benchmark script in `scripts/`
2. Documentation in `docs/BENCHMARK_GUIDE.md`
3. Example results in `results/examples/`

---

## References

- [NIST FRVT Benchmark Suite](https://nvlpubs.nist.gov/nistpubs/papers/2022/nist.sp.800-188.pdf)
- [ISO/IEC 19989 Face Recognition](https://www.iso.org/standard/66481.html)
- [ArcFace: Additive Angular Margin Loss](https://arxiv.org/abs/1801.07698)
- [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)

---

## Support & Issues

For issues running benchmarks:

1. Check [logs/benchmark_*.log](../logs/)
2. Open [GitHub Issue](https://github.com/ShubhamPatra/attendance_system/issues)
3. Include benchmark version, seed, and error output

