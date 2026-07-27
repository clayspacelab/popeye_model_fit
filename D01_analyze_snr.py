"""
D01_analyze_snr.py
==================
Task-band Fractional Spectral Power (TFSP) — SNR Diagnostics

OVERVIEW
--------
Unified diagnostic script for characterizing pRF data quality using
Task-band Fractional Spectral Power (TFSP).  Supports two modes:

  --mode subject    (default) Real surface (GIFTI) data for a subject.
                    Per-hemisphere: sample signal traces, FFT inspection,
                    TFSP histogram, quantile signal profiles, and correlation
                    with existing fit estimates if present.

  --mode simulation Simulated volumetric data from S01_simulate_prf.py.
                    Correlates TFSP with pRF fit R² from S02 outputs.
                    Outputs scatter, regression, quintile bar, summary panel.

MEASURE: Task-band Fractional Spectral Power (TFSP)
------------------------------------------------------
TFSP is a stimulus-aware extension of fALFF (Zou et al. 2008, Brain Research):

    TFSP = Σ |X(f)| for f in [f_min, f_max]
           ─────────────────────────────────────
           Σ |X(f)| for all f > f_drift

where |X(f)| is the FFT amplitude (not power) at frequency f.

Design decisions vs. naïve relative spectral power:

  1. AMPLITUDE not power: |FFT|, not |FFT|², following fALFF convention.
     Power over-weights large spectral bins and is less robust to
     physiological artifacts.  Amplitude weighting is more uniform.

  2. STIMULUS-DERIVED band [f_min, f_max]: Derived from the bar sweep
     cycle period rather than a fixed resting-state convention.
       f_fundamental = 1 / sweep_period_s   (e.g. 1/25 = 0.04 Hz)
       f_min         = f_fundamental        (fundamental)
       f_max         = 4 * f_fundamental    (up to 4th harmonic)
     Adapts automatically to any experiment design.

  3. DRIFT FLOOR f_drift: Frequencies below f_fundamental / 2 are
     excluded from both numerator and denominator, preventing scanner
     drift from artificially inflating the measure.

WHY TFSP PREDICTS R²
---------------------
Task stimuli (bar sweeps) concentrate BOLD power in the low-frequency
band set by sweep periodicity.  Broadband noise (thermal, physiological)
distributes uniformly.  Therefore:

    High signal-to-noise → high TFSP → pRF model fits well → high R²

USAGE
-----
  # Subject mode (real surface data)
  python D01_analyze_snr.py --subject MAM0606
  python D01_analyze_snr.py --subject MAM0606 --sweep-period 25.0

  # Simulation mode (S01 pkl + S02 .npy)
  python D01_analyze_snr.py --mode simulation --grid-size 50
  python D01_analyze_snr.py --mode simulation --grid-size 50 --sweep-period 25.0

OUTPUTS
-------
  Subject mode  → {popeyeFitDir}/snrtesting/
      signal_fft_sample_{hemi}.png
      tfsp_histogram_{hemi}.png
      quantiles_signals_{hemi}.png
      tfsp_vs_r2_{hemi}.png         (if fit estimates found)

  Simulation mode → Simulation/figures/testing_snr/
      tfsp_vs_r2_{tag}.png
      tfsp_summary_panel_{tag}.png
"""

import os
import sys
import ctypes
import pickle
import argparse

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from H01_config import DEFAULT_PARAMS, GRID_DEFAULTS, set_paths
from H02_dataloader import load_surface_data
from H03_fit_utils import remove_trend, set_dark_theme


# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_SWEEP_PERIOD_S = 25.0   # bar sweep cycle period (seconds)
DEFAULT_TR             = 1.3    # seconds


# ─── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Task-band Fractional Spectral Power — SNR Diagnostics',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--mode', choices=['subject', 'simulation'],
                        default='subject',
                        help='subject: real GIFTI surface data; '
                             'simulation: pkl data from S01 (default: subject)')
    parser.add_argument('--subject', '-s', default='MAM0606',
                        help='Subject ID (subject mode only, default: MAM0606)')
    parser.add_argument('--grid-size', type=int, default=GRID_DEFAULTS['Ns'],
                        help=f'Ns used in S02 (simulation mode, default: {GRID_DEFAULTS["Ns"]})')
    parser.add_argument('--sweep-period', type=float, default=DEFAULT_SWEEP_PERIOD_S,
                        help=f'Bar sweep cycle period in seconds '
                             f'(default: {DEFAULT_SWEEP_PERIOD_S})')
    parser.add_argument('--tr', type=float, default=DEFAULT_TR,
                        help=f'Repetition time in seconds (default: {DEFAULT_TR})')

    return parser.parse_args()


# ─── TFSP core computation ────────────────────────────────────────────────────

def compute_tfsp(timeseries, tr, sweep_period_s):
    """
    Compute Task-band Fractional Spectral Power (TFSP) per voxel/vertex.

    Parameters
    ----------
    timeseries : ndarray (N, T)
        Detrended timeseries (N voxels/vertices, T timepoints).
    tr : float
        Repetition time in seconds.
    sweep_period_s : float
        Bar sweep cycle period in seconds; sets the task frequency band.

    Returns
    -------
    tfsp : ndarray (N,)  — values in [0, 1]
    f_min, f_max, f_drift : float — band edges and drift floor (Hz)
    """
    N, T = timeseries.shape

    f_fundamental = 1.0 / sweep_period_s
    f_min         = f_fundamental
    f_max         = min(4.0 * f_fundamental, 0.5 / tr)   # clamp to Nyquist
    f_drift       = f_fundamental / 2.0

    # Per-voxel z-score before FFT (removes amplitude differences across voxels)
    means = timeseries.mean(axis=1, keepdims=True)
    stds  = timeseries.std(axis=1,  keepdims=True)
    stds[stds == 0] = 1.0
    ts_norm = (timeseries - means) / stds

    data_fft = np.fft.rfft(ts_norm, axis=1)
    freqs    = np.fft.rfftfreq(T, d=tr)
    amp      = np.abs(data_fft)                           # amplitude, not power

    task_mask  = (freqs >= f_min) & (freqs <= f_max)
    total_mask = freqs >  f_drift

    numerator   = amp[:, task_mask].sum(axis=1)
    denominator = amp[:, total_mask].sum(axis=1)
    denominator[denominator == 0] = 1.0

    return numerator / denominator, f_min, f_max, f_drift


# ─── Shared plotting helpers ──────────────────────────────────────────────────

def _scatter_tfsp_vs_r2(ax, tfsp, r2, f_min, f_max, label_suffix=''):
    mask = np.isfinite(tfsp) & np.isfinite(r2)
    xs, ys = tfsp[mask], r2[mask]
    ax.scatter(xs, ys, s=4, alpha=0.35, color='#00e5ff', edgecolors='none',
               rasterized=True)
    if len(xs) > 2:
        r, p = stats.pearsonr(xs, ys)
        rho, p_sp = stats.spearmanr(xs, ys)
        m, b = np.polyfit(xs, ys, 1)
        xl = np.array([xs.min(), xs.max()])
        ax.plot(xl, m * xl + b, '--', color='#ff4081', linewidth=1.5)
        ax.set_title(
            f'TFSP vs R²{label_suffix}\n'
            f'Pearson r={r:.3f} (p={p:.2e})   Spearman ρ={rho:.3f} (p={p_sp:.2e})'
        )
    else:
        ax.set_title(f'TFSP vs R²{label_suffix}')
    ax.set_xlabel(f'TFSP  [{f_min:.3f}–{f_max:.3f} Hz]')
    ax.set_ylabel('Model R²')
    ax.grid(True, alpha=0.3)


def _quintile_bar(ax, tfsp, r2, n_q=5):
    xs, ys = tfsp, r2
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    edges = np.percentile(xs, np.linspace(0, 100, n_q + 1))
    means, stds, labels = [], [], []
    for i in range(n_q):
        sel = ys[(xs >= edges[i]) & (xs <= edges[i + 1])]
        means.append(sel.mean())
        stds.append(sel.std())
        labels.append(f'Q{i+1}\n[{edges[i]:.2f}–{edges[i+1]:.2f}]')
    means, stds = np.array(means), np.array(stds)
    xpos = np.arange(n_q)
    ax.bar(xpos, means, color='#00e5ff', alpha=0.8, edgecolor='none')
    ax.errorbar(xpos, means, yerr=stds, fmt='none',
                color='#ff4081', capsize=4, linewidth=1.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel('TFSP Quintile')
    ax.set_ylabel('Mean R² ± std')
    ax.set_title('Mean R² by TFSP Quintile')
    ax.grid(True, axis='y', alpha=0.3)


# ─── Subject mode ─────────────────────────────────────────────────────────────

def run_subject_mode(args):
    subj_id = args.subject
    tr      = args.tr
    sweep_p = args.sweep_period

    print(f"=== D01 Subject Mode: {subj_id} ===")
    p, funcFiles = set_paths(subj_id, data_format='surface')

    out_dir = os.path.join(p['popeyeFitDir'], 'snrtesting')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    print("Loading surface runs and averaging...")
    leftDataOrig, rightDataOrig, tr_length, nTRs = load_surface_data(p, funcFiles)
    # Use scanned TR if not overridden
    if args.tr == DEFAULT_TR and tr_length is not None:
        tr = tr_length

    print("Detrending...")
    leftDet  = remove_trend(leftDataOrig,  method='all')
    rightDet = remove_trend(rightDataOrig, method='all')

    hemispheres = {
        'left':  (leftDataOrig,  leftDet),
        'right': (rightDataOrig, rightDet),
    }

    set_dark_theme()

    for hemi, (data_orig, data_det) in hemispheres.items():
        print(f"\n── {hemi.upper()} hemisphere ──")
        n_vtx = data_det.shape[0]

        # Discard first 5 TRs (scanner stabilisation)
        ts = data_det[:, 5:]

        # ── 1. Compute TFSP ────────────────────────────────────────────────
        print("Computing TFSP...")
        tfsp, f_min, f_max, f_drift = compute_tfsp(ts, tr, sweep_p)
        print(f"  Band [{f_min:.4f}, {f_max:.4f}] Hz  |  "
              f"drift floor > {f_drift:.4f} Hz  |  "
              f"TFSP median={np.median(tfsp):.3f}")

        # ── 2. Sample signal + amplitude spectrum ─────────────────────────
        print("Plotting sample signal + spectrum...")
        means_ts = ts.mean(axis=1, keepdims=True)
        stds_ts  = ts.std(axis=1,  keepdims=True)
        stds_ts[stds_ts == 0] = 1.0
        ts_norm  = (ts - means_ts) / stds_ts
        data_fft = np.fft.rfft(ts_norm, axis=1)
        freqs    = np.fft.rfftfreq(ts.shape[1], d=tr)

        vtx = np.random.randint(0, n_vtx)
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))
        axs[0].plot(data_det[vtx, 5:], color='#00e5ff', linewidth=0.8)
        axs[0].set_title(f'Detrended Signal (Vertex {vtx})')
        axs[0].set_xlabel('TR')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(freqs[1:], np.abs(data_fft[vtx, 1:]), color='#00e5ff', linewidth=0.8)
        axs[1].axvspan(f_min, f_max, alpha=0.2, color='#ff9800', label='Task band')
        axs[1].axvline(f_drift, color='#ff4081', linestyle='--', linewidth=1,
                       label=f'Drift floor ({f_drift:.4f} Hz)')
        axs[1].legend(fontsize=8, framealpha=0.3)
        axs[1].set_title(f'Amplitude Spectrum (Vertex {vtx})')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('|X(f)|')
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'signal_spectrum_sample_{hemi}.png'), dpi=150)
        plt.close(fig)

        # ── 3. TFSP histogram ─────────────────────────────────────────────
        print("Plotting TFSP histogram...")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(tfsp, bins=200, color='#00e5ff', edgecolor='none', alpha=0.85)
        ax.axvline(np.median(tfsp), color='#ff4081', linestyle='--', linewidth=1.3,
                   label=f'median={np.median(tfsp):.3f}')
        ax.legend(fontsize=10, framealpha=0.3)
        ax.set_title(f'TFSP Distribution — {hemi.capitalize()} Hemisphere')
        ax.set_xlabel(f'Task-band Fractional Spectral Power  [{f_min:.3f}–{f_max:.3f} Hz]')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'tfsp_histogram_{hemi}.png'), dpi=150)
        plt.close(fig)

        # ── 4. Quantile signal profiles ────────────────────────────────────
        print("Plotting quantile signal profiles...")
        n_quantiles = 4
        nVtx        = 5
        edges = np.percentile(tfsp, np.linspace(0, 100, n_quantiles + 1))
        quantile_indices = []
        for i in range(n_quantiles):
            if i < n_quantiles - 1:
                idx = np.where((tfsp >= edges[i]) & (tfsp < edges[i + 1]))[0]
            else:
                idx = np.where(tfsp >= edges[i])[0]
            quantile_indices.append(idx)

        fig, axs = plt.subplots(n_quantiles, nVtx,
                                figsize=(5 * nVtx, 3 * n_quantiles), squeeze=False)
        for q in range(n_quantiles):
            for i in range(nVtx):
                ax = axs[q, i]
                if len(quantile_indices[q]) == 0:
                    ax.set_visible(False)
                    continue
                vtx = np.random.choice(quantile_indices[q])
                orig_z = (data_orig[vtx] - data_orig[vtx].mean()) / (data_orig[vtx].std() or 1)
                ax.plot(orig_z,         color='white',  alpha=0.5,  linewidth=0.7, label='Orig')
                ax.plot(data_det[vtx],  color='#ff4081', linewidth=0.7, label='Detrended')
                ax.set_title(f'Q{q+1} vtx={vtx}  TFSP={tfsp[vtx]:.2f}', fontsize=8)
                if i == 0:
                    ax.legend(fontsize=7, framealpha=0.3)
                ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'quantiles_signals_{hemi}.png'), dpi=150)
        plt.close(fig)

        # ── 5. Correlate with fit estimates if present ─────────────────────
        fit_paths = [
            os.path.join(p['popeyeFitDir'], 'fitEstimatesOrig',
                         f'RF_ss5_gFit_popeye_{hemi}.func.gii'),
            os.path.join(p['popeyeFitDir'], 'fitEstimates',
                         f'RF_ss5_gFit_popeye_{hemi}.func.gii'),
        ]
        fit_path = next((fp for fp in fit_paths if os.path.exists(fp)), None)

        if fit_path:
            print(f"Loading fit estimates: {fit_path}")
            gii   = nib.load(fit_path)
            r2    = np.array([x.data for x in gii.darrays]).T[:, 1]

            n_fit = len(r2)
            tfsp_aligned = tfsp[:n_fit] if n_fit < len(tfsp) else tfsp

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            _scatter_tfsp_vs_r2(axes[0], tfsp_aligned, r2, f_min, f_max,
                                label_suffix=f' — {hemi.capitalize()}')
            _quintile_bar(axes[1], tfsp_aligned, r2)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'tfsp_vs_r2_{hemi}.png'), dpi=200)
            plt.close(fig)
            print(f"  Saved: tfsp_vs_r2_{hemi}.png")
        else:
            print(f"  No fit estimates found for {hemi} — skipping correlation plot.")

    print("\nSubject mode complete.")


# ─── Simulation mode ──────────────────────────────────────────────────────────

def run_simulation_mode(args):
    Ns      = args.grid_size
    tr      = args.tr
    sweep_p = args.sweep_period

    params = dict(DEFAULT_PARAMS)
    params['subjID'] = 'JC'
    p, _ = set_paths(params['subjID'], data_format='volumetric')

    sim_dir     = os.path.join(p['pRF_data'], 'Simulation')
    sim_fit_dir = os.path.join(sim_dir, 'popeyeFit')
    out_dir     = os.path.join(sim_dir, 'figures', 'testing_snr')
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== D01 Simulation Mode (Ns={Ns}) ===")
    print(f"TR={tr}s  |  Sweep period={sweep_p}s  |  Output: {out_dir}")

    # Load simulated timeseries
    vox_path = os.path.join(sim_dir, 'simulatedVoxels.pkl')
    if not os.path.exists(vox_path):
        print(f"ERROR: {vox_path} not found. Run S01_simulate_prf.py first.")
        sys.exit(1)

    print("Loading simulated voxels...")
    with open(vox_path, 'rb') as fh:
        scan_data = pickle.load(fh)
    print(f"  {scan_data.shape[0]:,} voxels x {scan_data.shape[1]} TRs")

    print("Detrending...")
    scan_det = remove_trend(scan_data, method='all')

    print("Computing TFSP...")
    tfsp, f_min, f_max, f_drift = compute_tfsp(scan_det, tr, sweep_p)
    print(f"  Band [{f_min:.4f}, {f_max:.4f}] Hz  |  "
          f"drift floor > {f_drift:.4f} Hz")
    print(f"  TFSP  min={tfsp.min():.3f}  max={tfsp.max():.3f}  "
          f"mean={tfsp.mean():.3f}  median={np.median(tfsp):.3f}")

    # Load R²
    ffit = os.path.join(sim_fit_dir, f'RF_ss5_fFit_popeye_Ns{Ns}.npy')
    gfit = os.path.join(sim_fit_dir, f'RF_ss5_gFit_popeye_Ns{Ns}.npy')
    if os.path.exists(ffit):
        print(f"Loading final-fit: {ffit}")
        fit_data = np.load(ffit)
        tag      = f'_ffit_Ns{Ns}'
    elif os.path.exists(gfit):
        print(f"Final-fit not found; using grid-fit: {gfit}")
        fit_data = np.load(gfit)
        tag      = f'_gfit_Ns{Ns}'
    else:
        print(f"ERROR: No fit files for Ns={Ns} in {sim_fit_dir}")
        print(f"Run:  python S02_run_simulation_fit.py --grid-size {Ns}")
        sys.exit(1)

    r2_vals = fit_data[:, 1]   # CSS column 1 = R²
    n_fit   = fit_data.shape[0]
    if n_fit < scan_data.shape[0]:
        print(f"  Trimming TFSP to {n_fit:,} voxels")
        tfsp = tfsp[:n_fit]
    print(f"  R²   min={r2_vals.min():.3f}  max={r2_vals.max():.3f}  "
          f"mean={r2_vals.mean():.3f}")

    set_dark_theme()

    # ── Three-panel: scatter | TFSP hist | R² hist ────────────────────────
    print("Plotting scatter panel...")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    _scatter_tfsp_vs_r2(axes[0], tfsp, r2_vals, f_min, f_max)

    axes[1].hist(tfsp[np.isfinite(tfsp)], bins=80,
                 color='#ff9800', edgecolor='none', alpha=0.85)
    axes[1].axvline(np.median(tfsp), color='#ff4081', linestyle='--', linewidth=1.2,
                    label=f'median={np.median(tfsp):.3f}')
    axes[1].legend(fontsize=9, framealpha=0.3)
    axes[1].set_xlabel('TFSP')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'TFSP Distribution\n(drift floor > {f_drift:.4f} Hz)')
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(r2_vals[np.isfinite(r2_vals)], bins=80,
                 color='#00e5ff', edgecolor='none', alpha=0.85)
    axes[2].axvline(np.median(r2_vals), color='#ff4081', linestyle='--', linewidth=1.2,
                    label=f'median={np.median(r2_vals):.3f}')
    axes[2].legend(fontsize=9, framealpha=0.3)
    axes[2].set_xlabel('Model R²')
    axes[2].set_ylabel('Count')
    axes[2].set_title('R² Distribution')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Task-band Fractional Spectral Power vs pRF Model R²',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'tfsp_vs_r2{tag}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")

    # ── Summary panel: scatter | quintile bar | TFSP hist | R² hist ──────
    print("Plotting summary panel...")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    _scatter_tfsp_vs_r2(axs[0, 0], tfsp, r2_vals, f_min, f_max)
    _quintile_bar(axs[0, 1], tfsp, r2_vals)

    axs[1, 0].hist(tfsp[np.isfinite(tfsp)], bins=80,
                   color='#ff9800', edgecolor='none', alpha=0.85)
    axs[1, 0].axvline(np.median(tfsp), color='#ff4081', linestyle='--', linewidth=1.2,
                      label=f'median={np.median(tfsp):.3f}')
    axs[1, 0].legend(fontsize=9, framealpha=0.3)
    axs[1, 0].set_xlabel('Task-band Fractional Spectral Power')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].set_title(f'TFSP Distribution  (drift floor > {f_drift:.4f} Hz)')
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].hist(r2_vals[np.isfinite(r2_vals)], bins=80,
                   color='#00e5ff', edgecolor='none', alpha=0.85)
    axs[1, 1].axvline(np.median(r2_vals), color='#ff4081', linestyle='--', linewidth=1.2,
                      label=f'median={np.median(r2_vals):.3f}')
    axs[1, 1].legend(fontsize=9, framealpha=0.3)
    axs[1, 1].set_xlabel('Model R²')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].set_title('R² Distribution from Fit')
    axs[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Task-band Fractional Spectral Power — Summary',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'tfsp_summary_panel{tag}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")

    print("\nSimulation mode complete.")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.mode == 'subject':
        run_subject_mode(args)
    else:
        run_simulation_mode(args)


if __name__ == '__main__':
    main()
