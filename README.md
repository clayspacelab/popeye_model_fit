# CSS pRF Model Fitting Pipeline (`popeye_model_fit`)

Population Receptive Field (pRF) model fitting pipeline using the **Compressive Spatial Summation (CSS)** model via [popeye](https://github.com/kdesimone/popeye). Fits 2D Gaussian spatial receptive fields with non-linear spatial summation to fMRI time-series data.

Supports both **volumetric (NIfTI)** and **surface (GIFTI)** fMRI data formats, with CPU multiprocessing and optional **GPU acceleration** via CuPy.

---

## Codebase Organization & Naming Conventions

The codebase follows a clear prefixed structure:

* **`H01` – `H06` (Helper Modules)**: Core library modules. Imported by main scripts; never executed directly.
* **`01_run_pipeline.py` (Main Pipeline)**: Primary entry point for fitting real subject data.
* **`S01` – `S02` (Simulation Tools)**: Synthetic data generation and model validation.
* **`D01` – `D02` (Diagnostics)**: Data quality analysis and SNR characterization tools.
* **`notebooks/`**: Interactive exploration and debugging notebooks.
* **`deprecated/`**: Legacy files, volume-only fitters, and superseded prototypes.
* **`deprecated_hpc/`**: NYU Greene HPC environment scripts and containers.

---

## File Reference

| File | Role & Description |
| :--- | :--- |
| **`H01_config.py`** | Central configuration: host detection (`vader`, `local_mac`, etc.), path generation, `GRID_DEFAULTS` (default `Ns=50`), CSS field names, and `get_gridfit_path(p, Ns)` helper. |
| **`H02_dataloader.py`** | Unified data ingestion for volumetric (NIfTI) and surface (GIFTI) data, stimulus loading via popeye `VisualStimulus`, and file I/O. |
| **`H03_fit_utils.py`** | Shared utilities: polynomial detrending, percent signal change, eccentricity-based grid constraints, and `set_dark_theme()` for all figures. |
| **`H04_grid_predict.py`** | Generates predicted BOLD timeseries for all `(x, y, σ, n)` grid points in parallel (CPU pool). Results cached as `gridfit_{Ns}.npy`. |
| **`H05_grid_fit.py`** | Coarse grid search via vectorized OLS. **CPU**: single matmul per voxel batch. **GPU** (CuPy): dual-tiled matrix operations with dynamic memory management — tile sizes adapt to free VRAM at runtime to avoid OOM. |
| **`H06_final_fit.py`** | Fine-grained refinement. **CPU**: L-BFGS-B via `scipy.optimize.minimize` with Pool initializer. **GPU** (CuPy): batched Adam optimizer for all voxels simultaneously. |
| **`01_run_pipeline.py`** | **Primary CLI entry point.** Orchestrates: stimulus loading → grid prediction → grid fit → final fit → output. |
| **`S01_simulate_prf.py`** | Generates synthetic pRF timeseries with ground-truth CSS parameters, additive noise, baseline, and linear trend. Supports GPU batch forward model (CuPy). Default: 100,000 voxels. |
| **`S02_run_simulation_fit.py`** | Fits simulated data from S01 and validates against ground truth. Comparison figures show identity scatter for spatial parameters (x, y, σ, n, θ, ρ) and **distributions** for R² and beta. All outputs labeled with `Ns`. |
| **`D01_analyze_snr.py`** | Computes per-vertex SNR metrics on real surface data: FFT power spectra, relative low-frequency power histograms, and quantile-stratified signal profiles. Correlates with existing grid-fit estimates if present. |
| **`D02_snr_vs_r2_simulation.py`** | Computes **Task-band Fractional Spectral Power (TFSP)** on simulated timeseries and correlates with pRF fit R². TFSP band is stimulus-derived (sweep period → fundamental frequency), uses FFT amplitude (not power), and applies a drift floor. Outputs scatter, regression, and summary panel figures. |

---

## Quick Start & Usage

### 1. Main Pipeline (`01_run_pipeline.py`)

```bash
# Volumetric (NIfTI) fitting — default
python 01_run_pipeline.py --subject MAM0606

# Surface (GIFTI) fitting
python 01_run_pipeline.py --subject MAM0606 --data-format surface

# GPU-accelerated (requires CuPy + CUDA)
python 01_run_pipeline.py --subject MAM0606 --use-gpu

# Custom grid resolution, skip final refinement
python 01_run_pipeline.py --subject MAM0606 --grid-size 50 --skip-final-fit
```

**Arguments:**
* `--subject` / `-s` *(required)*: Subject identifier (e.g., `MAM0606`).
* `--data-format`: `volumetric` (default) or `surface`.
* `--use-gpu`: Enable CuPy GPU acceleration for grid fit and final fit.
* `--grid-size`: Grid density `Ns` (default: `50`). Grid predictions are cached per `Ns` as `gridfit_{Ns}.npy`.
* `--skip-final-fit`: Run coarse grid fit only; skip gradient-descent refinement.
* `--hemisphere`: `both` (default), `left`, or `right` (surface mode only).

---

### 2. Simulation & Validation (`S01` → `S02`)

```bash
# Step 1: Generate 100,000 synthetic voxels (GPU-accelerated forward model)
python S01_simulate_prf.py --n-voxels 100000 --use-gpu

# Step 2: Fit and validate against ground truth
python S02_run_simulation_fit.py --grid-size 50 --use-gpu

# Skip final fit (grid only)
python S02_run_simulation_fit.py --grid-size 50 --use-gpu --skip-final-fit
```

Output files are labeled with `Ns` (e.g., `RF_ss5_gFit_popeye_Ns50.npy`) so runs with different grid sizes coexist without overwriting.

Comparison figures show:
* **Identity scatter** (ground truth vs fitted) for: `theta, rho, sigma, n, x, y`
* **Distribution histogram** (fitted values) for: `R²` and `beta`

All figures use dark mode (black background, cyan/magenta accents).

---

### 3. SNR Diagnostics (`D01` & `D02`)

```bash
# D01 — SNR analysis on real surface data
python D01_analyze_snr.py MAM0606

# D02 — Task-band FSP vs R² on simulation data
python D02_snr_vs_r2_simulation.py --grid-size 50

# D02 with custom sweep period
python D02_snr_vs_r2_simulation.py --grid-size 50 --sweep-period 25.0
```

**D02 outputs** are saved to `Simulation/figures/testing_snr/`:
* `tfsp_vs_r2_{tag}.png` — scatter + regression + marginal histograms
* `tfsp_summary_panel_{tag}.png` — scatter | quintile bar | TFSP hist | R² hist

---

## Parameter Outputs

The pipeline estimates 9 parameters per voxel/vertex:

| Index | Name | Description |
| :---: | :--- | :--- |
| 0 | `theta` | Polar angle (radians, [0, 2π]) |
| 1 | `r2` | Variance explained (R²) |
| 2 | `rho` | Eccentricity (degrees of visual angle) |
| 3 | `sigma` | RF size / dispersion (σ) |
| 4 | `n` | CSS compressive exponent |
| 5 | `x` | Horizontal RF centre (degrees) |
| 6 | `y` | Vertical RF centre (degrees) |
| 7 | `beta` | Amplitude gain |
| 8 | `baseline` | Signal DC offset / intercept |

---

## GPU Acceleration

Both `H05_grid_fit.py` and `H06_final_fit.py` support `--use-gpu` via CuPy.

**Grid fit (H05) GPU strategy:**
- Loads all grid predictions and timeseries to VRAM first, then flushes the memory pool
- Queries free VRAM *after* loading to get an accurate budget
- Tiles both voxels (`B_vox`) and grid points (`G_chunk`) to fit within budget
- Uses in-place matmul + `cp.maximum` clipping instead of boolean mask assignment (avoids hidden CuPy prefix-scan allocations that cause OOM)

**Final fit (H06) GPU strategy:**
- Batched Adam optimizer; all voxels optimized simultaneously in CuPy
- Falls back to CPU L-BFGS-B if CuPy is unavailable

Tested on: NVIDIA Titan Xp (12 GB VRAM), CUDA 12.2.

---

## Grid Prediction Caching

Grid predictions are expensive (minutes for Ns=50, ~3 min for Ns=100). They are cached per `Ns` in `Stimuli/`:

```
pRF_data/Stimuli/
    gridfit_50.npy     ← Ns=50 cache
    gridfit_100.npy    ← Ns=100 cache
```

On re-run, if the cache exists the generation step is skipped automatically.

---

## Compute Environment Setup

```bash
# Activate environment (Vader / lab cluster)
conda activate prf_fitter

# Verify GPU is available
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

**Core dependencies:**
* Python 3.9+
* `numpy`, `scipy`, `nibabel`, `matplotlib`, `tqdm`
* `popeye` (pRF modeling library)
* `cupy` *(optional — for GPU acceleration)*
