# KPFuse v2 Smoothing & Stability Update Log

This document summarizes the modifications made to `kpfuse_v2.py` to reduce patchy fusion artifacts and improve output smoothness while keeping detail retention controllable.

## 1) Priority-map smoothing (high impact)

### What changed
- Added helper utilities:
  - `_as_odd_kernel(...)`
  - `smooth_spatial_map(...)`
- Added configurable smoothing for the exclusive IR priority map (`ir_priority_excl`) in `FusionNet._compute_exclusive_ir_weight(...)`.

### Why it helps
- The exclusivity signal is based on high-frequency gradient and keypoint differences. Directly using it causes pixel-level hard switching and visible patch boundaries.
- Smoothing turns abrupt local switches into spatially coherent transitions.

### New knobs
- `--excl-smooth-mode {gaussian,box}` (default: `gaussian`)
- `--excl-smooth-kernel` (default: `7`)
- `--excl-smooth-sigma` (default: `1.6`)
- `--excl-smooth-passes` (default: `1`)


## 2) Keypoint-map smoothing before gating/attention bias

### What changed
- Smoothed `vis_kmap` and `ir_kmap` after interpolation and before:
  - feature gating multipliers
  - attention bias injection

### Why it helps
- Raw/sparse keypoint response maps can produce localized hotspots, which can create uneven texture islands in fused output.
- Light smoothing stabilizes regional emphasis without removing keypoint guidance.

### New knobs
- `--kmap-smooth-mode {gaussian,box}` (default: `gaussian`)
- `--kmap-smooth-kernel` (default: `3`)
- `--kmap-smooth-sigma` (default: `0.9`)
- `--kmap-smooth-passes` (default: `1`)


## 3) Safer and smoother luminance gain blending

### What changed
- Updated gain computation in `_adaptive_blend(...)`:
  - denominator epsilon is configurable and larger by default
  - tighter gain clamp range by default
  - gain map is spatially smoothed
  - gain influence is mixed with neutral gain via a strength factor

### Why it helps
- Previous local gain could spike strongly in dark VIS regions, causing blotchy amplification.
- Smoothing + bounded + blended gain reduces abrupt local luminance/color jumps.

### New knobs
- `--gain-eps` (default: `1e-3`)
- `--gain-min` (default: `0.4`)
- `--gain-max` (default: `2.2`)
- `--gain-smooth-mode {gaussian,box}` (default: `gaussian`)
- `--gain-smooth-kernel` (default: `5`)
- `--gain-smooth-sigma` (default: `1.1`)
- `--gain-smooth-passes` (default: `1`)
- `--gain-strength` in `[0,1]` (default: `0.7`)


## 4) Less aggressive defaults for gating and attention bias

### What changed
- Reduced defaults:
  - `--gate-alpha-vis`: `0.25 -> 0.10`
  - `--gate-alpha-ir`: `0.35 -> 0.15`
  - `--attn-bias-gamma`: `0.20 -> 0.10`

### Why it helps
- Strong spatial gating/bias can over-concentrate detail in hotspots and create patchy nonuniformity.
- Lower defaults provide smoother, more globally consistent fusion.


## 5) Slightly more learned RGB contribution

### What changed
- Increased defaults:
  - `--luma-pred-mix`: `0.06 -> 0.12`
  - `--pred-rgb-mix`: `0.04 -> 0.10`

### Why it helps
- Final output was dominated by analytic blend maps, making artifacts from noisy maps more visible.
- Slightly increasing learned prediction contribution improves continuity and texture coherence.


## 6) Native-resolution padding improvement

### What changed
- In `_pad_image_and_mask(...)`, changed image padding from zero padding to `replicate` padding.

### Why it helps
- Zero padding introduces strong artificial borders that can leak into feature extraction and attention.
- Replicate padding reduces edge discontinuities and border artifacts.


## 7) Alignment default for 4-downsample architecture

### What changed
- Updated default `--pad-multiple` from `8` to `16`.

### Why it helps
- With 4 stride-2 downsamples, 16-aligned native-resolution dimensions reduce shape correction/interpolation edge cases.


## 8) Argument validation updates

Added CLI validation for new smoothing and gain parameters:
- kernels >= 1
- sigmas > 0
- passes >= 0
- gain-eps > 0
- gain-max >= gain-min
- gain-strength in [0, 1]


## 9) Backward compatibility and control

- The architecture remains compatible with the existing training/inference flow.
- All smoothing and gain behavior is tunable from CLI.
- You can reduce/disable smoothing by setting kernel to 1 or passes to 0.


## Suggested stable baseline command

Use defaults first, then tune:

```bash
python kpfuse_v2.py \
  --vis-dir /path/to/vi \
  --ir-dir /path/to/ir \
  --gt-dir /path/to/gt \
  --epochs 100 \
  --batch-size 8 \
  --size 512 \
  --native-res-train \
  --pad-multiple 16 \
  --ckpt best_kpfuse_v2_smooth.pth
```

## Optional stronger smoothing profile

```bash
python kpfuse_v2.py \
  --vis-dir /path/to/vi \
  --ir-dir /path/to/ir \
  --gt-dir /path/to/gt \
  --excl-smooth-kernel 9 \
  --excl-smooth-sigma 2.0 \
  --gain-smooth-kernel 7 \
  --gain-smooth-sigma 1.4 \
  --gain-strength 0.6
```
