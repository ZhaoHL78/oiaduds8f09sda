# Spherical Kikuchi Fine-Tune Pipeline

This note describes the single-pattern spherical correction implemented in
`spherical_finetune_pipeline.py`.

For flowcharts, see `SPHERICAL_FINETUNE_PIPELINE_VISUAL.md`.

## Goal

Place one experimental EDAX Kikuchi pattern onto a standard Kikuchi master
sphere, then refine only small physically meaningful residual errors:

- orientation residual: a small 3D rotation around the H5/OIM orientation
- spherical dilation: a small `radius_scale` applied as `PCz_eff = PCz / radius_scale`
- optional PC residual: small `PCx/PCy/PCz` offsets if explicitly enabled

The output is both diagnostic visualization and a pixel-to-sphere mapping that
can be reused for Kikuchi sphere reconstruction.

## Simplified Pipeline

1. Preprocess the Kikuchi pattern.
   - mask detector region
   - normalize intensity
   - remove smooth background
   - optionally enhance Kikuchi bands

2. Perform initial spherical calibration.
   - assume an initial PC, detector geometry, sample tilt, and orientation
   - map detector pixels to rays
   - place those rays on the crystal/master Kikuchi sphere

3. Fine tune the geometric parameters.
   - start with small rotation and `radius_scale`
   - release `PCx/PCy/PCz` only when residuals suggest a systematic PC error
   - optionally interpret `PCz/radius_scale` as detector distance or spherical
     expansion/compression

4. Diagnose residuals.
   - compare the experimental pattern with the master-sphere prediction sampled
     back onto the detector
   - use residual structure to decide whether PC, tilt, detector distance, or
     orientation is responsible

5. Invert refined geometry.
   - report refined PC
   - report effective detector distance / radius scale
   - report tilt bias and small orientation correction
   - save the final pixel-to-sphere mapping

## Outputs

The script writes:

- `00_raw_preprocess_mask.png`
- `01_initial_candidate_scores.png`
- `02_finetune_trace.png`
- `02_finetune_trace.csv`
- `03_sphere_overlay_initial_vs_refined.png`
- `04_detector_forward_residual.png`
- `05_patch_on_unit_sphere_3d.png`
- `refined_pixel_to_sphere_mapping.npz`
- `summary.json`
- optional `refined_pixel_to_sphere_mapping.csv`

## Recommended Defaults

For a first run, keep PC fixed and refine only rotation plus radius:

```powershell
python .\spherical_finetune_pipeline.py `
  --h5 path\to\ebsd.edaxh5 `
  --up2 path\to\map.up2 `
  --map-group "/path/to/OIM Map HighR" `
  --pattern-index 2661 `
  --master path\to\master_pattern.h5 `
  --output-dir outputs\spherical_finetune\idx_02661
```

If the residual image shows a systematic left/right or up/down drift, enable
small PC offsets:

```powershell
python .\spherical_finetune_pipeline.py `
  --h5 path\to\ebsd.edaxh5 `
  --up2 path\to\map.up2 `
  --map-group "/path/to/OIM Map HighR" `
  --pattern-index 2661 `
  --master path\to\master_pattern.h5 `
  --fit-pcxy --fit-pcz `
  --output-dir outputs\spherical_finetune\idx_02661_pc
```

Use `--global-iter 0` for a fast local-only test, or increase
`--global-iter` when the initial convention is uncertain.
