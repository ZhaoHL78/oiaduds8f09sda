# Spherical Kikuchi Calibration Pipeline

This is the simplified conceptual pipeline. The goal is not to show every
script output, but to show the main physical calibration logic.

## Publication Figure

Use `make_physical_spherical_pipeline_figure.py` to generate a publication-style
physics schematic:

```powershell
python .\make_physical_spherical_pipeline_figure.py
```

The generated files are:

- `outputs/article_figures/spherical_calibration_pipeline/physical_spherical_calibration_pipeline.svg`
- `outputs/article_figures/spherical_calibration_pipeline/physical_spherical_calibration_pipeline.pdf`
- `outputs/article_figures/spherical_calibration_pipeline/physical_spherical_calibration_pipeline.png`

## Main Idea

```mermaid
flowchart TD
    A["Input<br/>Kikuchi pattern + master sphere + initial geometry"] --> B["Preprocess<br/>mask, normalize, background correction, optional band enhancement"]
    B --> C["Initial spherical calibration<br/>use assumed PC, detector geometry, sample tilt, and orientation"]
    C --> D["Fine tune<br/>adjust PC, small rotation, radius / detector geometry parameters"]
    D --> E["Residual diagnosis<br/>compare experimental pattern with master-sphere prediction"]
    E --> F["Geometry inversion<br/>recover refined PC, detector distance, tilt, and orientation correction"]

    E -- "residual still structured" --> D
    E -- "residual random / acceptable" --> F
```

## Fine-Tune Core

```mermaid
flowchart LR
    A["Current geometry parameters<br/>PCx, PCy, PCz, tilt, small rotation, radius scale"] --> B["Pixel -> detector ray"]
    B --> C["Detector ray -> spherical coordinates"]
    C --> D["Sample master sphere"]
    D --> E["Calibration error<br/>intensity residual + band residual"]
    E --> F["Optimizer"]
    F --> A
    E --> G["Best refined geometry"]
```

## Parameter Meaning

```mermaid
flowchart TD
    A["Refined geometry"] --> B["PCx / PCy<br/>lateral pattern center shift"]
    A --> C["PCz / detector distance<br/>spherical expansion or compression"]
    A --> D["Detector / sample tilt<br/>systematic angular bias"]
    A --> E["Small orientation correction<br/>residual crystal-frame rotation"]
    A --> F["Residual map<br/>where calibration still fails"]
```

## Short Interpretation

- Preprocessing only makes the pattern comparable to the master sphere; it does
  not change the physical geometry.
- Initial spherical calibration is the first physics-based mapping from detector
  pixels to the Kikuchi sphere.
- Fine tune should be bounded and low-dimensional, otherwise it becomes image
  registration rather than EBSD geometry calibration.
- Residual diagnosis decides which parameter should be released next.
- The final result can be interpreted as inverted geometry: refined PC,
  detector distance, tilt bias, and orientation correction.
