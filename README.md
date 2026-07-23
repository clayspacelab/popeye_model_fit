# CSS pRF Model Fitting Pipeline (`popeye_model_fit`)

Population Receptive Field (pRF) model fitting pipeline using the **Compressive Spatial Summation (CSS)** model via [popeye](https://github.com/kdesimone/popeye). Fits 2D Gaussian spatial receptive fields with non-linear spatial summation to fMRI time-series data.

Supports both **volumetric (NIfTI)** and **surface (GIFTI)** fMRI data formats, with optional **GPU acceleration** via CuPy.

---

## Codebase Organization & Naming Conventions

The codebase follows a clear prefixed structure to distinguish helper modules, runnable pipeline scripts, simulation tools, and diagnostics:

* **`H01` – `H06` (Helper Modules)**: Core functionality modules. Imported by main scripts; never executed directly.
* **`01_run_pipeline.py` (Main Pipeline)**: The primary entry point for fitting real subject data.
* **`S01` – `S02` (Simulation Tools)**: Synthetic data generation and model validation scripts.
* **`D01_analyze_snr.py` (Diagnostics)**: Diagnostic tools for analyzing signal-to-noise ratio.
* **`notebooks/`**: Interactive exploration and debugging notebooks.
* **`deprecated/`**: Legacy files, volume-only fitters, and superseded prototypes.
* **`deprecated_hpc/`**: NYU Greene HPC environment scripts and containers.

---

## File Reference

| File / Folder | Role & Description |
| :--- | :--- |
| **`H01_config.py`** | Central configuration: paths, host environment detection (`vader`, `local_mac`, etc.), grid defaults, model parameters. |
| **`H02_dataloader.py`** | Unified data ingestion for volumetric (NIfTI) and surface (GIFTI) data, stimulus loading, and file I/O. |
| **`H03_fit_utils.py`** | Shared mathematical & preprocessing utilities (detrending, percent signal change, eccentricity grid constraints). |
| **`H04_grid_predict.py`** | Generates predicted BOLD time-series for 4D parameter search grids `(x, y, sigma, n)`. |
| **`H05_grid_fit.py`** | Coarse grid search for best-matching parameter estimates per voxel/vertex (CPU multiprocessing + vectorized GPU paths). |
| **`H06_final_fit.py`** | Fine-grained gradient-descent refinement (`scipy.optimize.minimize` SLSQP) with non-linear bounds. |
| **`01_run_pipeline.py`** | **Primary CLI entry point**. Orchestrates data loading, grid prediction, grid fit, and final optimization. |
| **`S01_simulate_prf.py`** | Generates synthetic pRF time-series with ground-truth parameters, noise, baseline, and linear trends. |
| **`S02_run_simulation_fit.py`** | Runs grid/final fitting on synthetic data and generates ground-truth comparison scatter plots. |
| **`D01_analyze_snr.py`** | Experimental diagnostic script evaluating signal-to-noise ratio and frequency power spectra. |

---

## Quick Start & Usage

### 1. Main Pipeline (`01_run_pipeline.py`)

Run model fitting on fMRI data for a given subject.

```bash
# Standard Volumetric (NIfTI) fitting (Default)
python 01_run_pipeline.py --subject MAM0606

# Surface (GIFTI) fitting
python 01_run_pipeline.py --subject MAM0606 --data-format surface

# Enable GPU acceleration (requires CuPy & CUDA)
python 01_run_pipeline.py --subject MAM0606 --use-gpu

# Custom grid resolution, skip final gradient descent
python 01_run_pipeline.py --subject MAM0606 --grid-size 50 --skip-final-fit
```

#### Command-Line Arguments:
* `--subject`, `-s` *(required)*: Subject identifier string (e.g., `MAM0606`).
* `--data-format`: `volumetric` (default) or `surface`.
* `--use-gpu`: Enable CuPy vectorized GPU grid-fit calculation.
* `--grid-size`: Density parameter $N_s$ for parameter grid search (default: `35`).
* `--skip-final-fit`: Run coarse grid-fit only; omit SLSQP optimization.
* `--hemisphere`: `both` (default), `left`, or `right` (applicable to surface mode).

---

### 2. Simulation & Validation (`S01` & `S02`)

Validate pipeline performance against known synthetic parameters:

```bash
# Step 1: Generate 1,000 synthetic voxels
python S01_simulate_prf.py --n-voxels 1000

# Step 2: Fit model to synthetic data & compare against ground truth
python S02_run_simulation_fit.py --n-voxels 100 --use-gpu
```

The validation script saves comparison scatter plots in `pRF_data/Simulation/figures/`.

---

## Parameter Outputs

The pipeline estimates 9 parameters per voxel/vertex:

1. **`theta`**: Polar angle (radians, $[0, 2\pi]$)
2. **`r2`**: Variance explained ($R^2$)
3. **`rho`**: Eccentricity (degrees of visual angle)
4. **`sigma`**: Receptive field size / dispersion ($\sigma$)
5. **`n`**: CSS exponent parameter ($n$)
6. **`x`**: Horizontal RF center (degrees)
7. **`y`**: Vertical RF center (degrees)
8. **`beta`**: Amplitude gain scaling factor
9. **`baseline`**: Signal DC offset / intercept

---

## Compute Environment Setup

### Vader Cluster
Ensure appropriate CUDA drivers and PyTorch/CuPy packages are loaded if using `--use-gpu`.
```bash
conda activate prf_fitter
```

### Dependencies
* Python 3.9+
* `numpy`, `scipy`, `nibabel`, `matplotlib`, `tqdm`
* `popeye` (pRF modeling package)
* `cupy` *(optional, for GPU acceleration)*
