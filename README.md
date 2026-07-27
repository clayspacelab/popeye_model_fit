# CSS pRF Model Fitting Pipeline (`popeye_model_fit`)

Population Receptive Field (pRF) model fitting pipeline using the **Compressive Spatial Summation (CSS)** model via [popeye](https://github.com/kdesimone/popeye). Fits 2D Gaussian spatial receptive fields with non-linear spatial summation to fMRI time-series data.

Supports both **volumetric (NIfTI)** and **surface (GIFTI)** fMRI data formats, with CPU multiprocessing and optional **GPU acceleration** via CuPy.

---

## Codebase Organization & Naming Conventions

The codebase follows a clear prefixed structure:

* **`H01` – `H06` (Helper Modules)**: Core library modules. Imported by main scripts; never executed directly.
* **`01_run_pipeline.py` (Main Pipeline)**: Primary entry point for fitting real subject data.
* **`S01` – `S03` (Simulation Tools)**: Synthetic data generation, model validation, and grid-size sweeps.
* **`D01` (Diagnostics)**: Data quality analysis and SNR characterization tools.
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
| **`S03_gridsize_sweep.py`** | Sweeps grid density `Ns` (default 10→100 by 10) over the S01 simulated data, running grid + final fit for each. Uses a **finer CSS exponent grid (10 values vs 4)** and defaults to **GPU**. Produces "accuracy vs Ns" (per-parameter correlation with ground truth), "R² vs Ns", and "runtime vs Ns" figures plus a `sweep_metrics.npz`. Grid-prediction caches are S03-specific (`gridfit_S03_Ns{Ns}_n{n_res}.npy`) so they never collide with S02's. |
| **`D01_analyze_snr.py`** | Unified diagnostic tool for TFSP analysis. `--mode subject` (default): real surface GIFTI data — sample signal traces, amplitude spectra, TFSP histogram, quantile signal profiles, and TFSP vs R² scatter when fit estimates are present. `--mode simulation`: loads pkl timeseries from S01 and correlates TFSP with pRF fit R² from S02, with scatter, quintile bar, and summary panel figures. |

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

#### Grid-size sweep (`S03`)

```bash
# Sweep Ns = 10, 20, ..., 100 (GPU by default), grid + final fit
python S03_gridsize_sweep.py

# Custom sweep, subset of voxels, CPU only
python S03_gridsize_sweep.py --grid-sizes 20 40 60 80 100 --n-voxels 5000 --no-gpu

# Grid fit only (skip final refinement)
python S03_gridsize_sweep.py --skip-final-fit
```

Outputs → `Simulation/figures/gridsize_sweep/`:
* `accuracy_vs_Ns.png` — Pearson correlation (fitted vs ground truth) vs Ns, one panel per parameter (`theta, rho, sigma, n, x, y`), grid & final fit.
* `r2_vs_Ns.png` — mean fitted R² as a function of Ns.
* `runtime_vs_Ns.png` — grid-fit / final-fit wall-clock time and constrained grid-point count vs Ns.
* `sweep_metrics.npz` — raw metrics for all Ns.

---

### 3. SNR Diagnostics (`D01` & `D02`)

```bash
# Subject mode — real surface data (default)
python D01_analyze_snr.py --subject MAM0606
python D01_analyze_snr.py --subject MAM0606 --sweep-period 25.0

# Simulation mode — S01 pkl timeseries + S02 .npy fit outputs
python D01_analyze_snr.py --mode simulation --grid-size 50
python D01_analyze_snr.py --mode simulation --grid-size 50 --sweep-period 25.0
```

**Subject mode outputs** → `{popeyeFitDir}/snrtesting/`:
* `signal_spectrum_sample_{hemi}.png` — sample signal + annotated amplitude spectrum
* `tfsp_histogram_{hemi}.png` — TFSP distribution per hemisphere
* `quantiles_signals_{hemi}.png` — signal traces stratified by TFSP quintile
* `tfsp_vs_r2_{hemi}.png` — TFSP vs R² scatter + quintile bar (if fit estimates found)

**Simulation mode outputs** → `Simulation/figures/testing_snr/`:
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
- **Auto-sized sub-batch**: fills available VRAM instead of a fixed 2000, keeping the GPU saturated and cutting Python/kernel-launch overhead (the dominant cost). Override with `sub_batch=`.
- **Fast FFT length**: the HRF-convolution FFT is zero-padded up to a 5-smooth length (identical result, faster cuFFT).
- **Cached-response gradient**: the exponent (`n`) finite-difference perturbation reuses the linear neural response — no RF rebuild or matmul — since only the nonlinearity depends on `n`.
- Tunable `n_iter` (Adam steps) and `lr` (learning rate); exposed via `S03_gridsize_sweep.py` (`--n-iter`, `--lr`, `--sub-batch`) so the iteration-count/accuracy tradeoff can be swept.
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
