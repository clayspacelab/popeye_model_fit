import os
import sys
import ctypes
import time
import numpy as np
import nibabel as nib

# Disable interactive plots for headless script execution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import custom modules from the repository
from H01_config import set_paths
from H02_dataloader import load_stimuli, load_surface_data
from H03_fit_utils import remove_trend, set_dark_theme

def main():
    # Allow subject ID to be passed as command line argument, default to MAM0606
    subj_id = sys.argv[1] if len(sys.argv) > 1 else 'MAM0606'
    print(f"Starting SNR analysis for subject: {subj_id}")

    # Initialize parameters
    params = {
        'subjID': subj_id,
        'viewingDistance': 83.5,
        'screenWidth': 36.2,
        'scaleFactor': 1,
        'resampleFactor': 1,
        'dtype': ctypes.c_int16
    }

    # Set paths and load average data
    p, funcFiles = set_paths(subj_id, data_format='surface')
    
    snr_dir = os.path.join(p['popeyeFitDir'], 'snrtesting')
    os.makedirs(snr_dir, exist_ok=True)
    print(f"Plots will be saved to: {snr_dir}")

    print("Loading surface runs and averaging...")
    leftDataOrig, rightDataOrig, tr_length, nTRs = load_surface_data(p, funcFiles)

    print("Detrending scan data...")
    leftDataOrigDetrended = remove_trend(leftDataOrig, method='all')
    rightDataOrigDetrended = remove_trend(rightDataOrig, method='all')

    hemispheres = {
        'left': (leftDataOrig, leftDataOrigDetrended),
        'right': (rightDataOrig, rightDataOrigDetrended)
    }

    set_dark_theme()

    for hemi, (data_orig, data_detrended) in hemispheres.items():
        print(f"\nProcessing {hemi} hemisphere...")
        # 1. Compute FFT of detrended data (discarding first 5 TRs for scanner stabilization)
        print("Computing FFT (vectorized)...")
        n_vertices = data_detrended.shape[0]
        thData = data_detrended[:, 5:]
        means = np.mean(thData, axis=1, keepdims=True)
        stds = np.std(thData, axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        thData_norm = (thData - means) / stds
        data_fft = np.fft.rfft(thData_norm, axis=1)
        
        # Calculate physical frequencies in Hz
        n_trs_used = thData.shape[1]
        freqs = np.fft.rfftfreq(n_trs_used, d=tr_length)

        # 2. Plot detrended signal & FFT for a random voxel/vertex
        print("Generating sample signal and FFT plot...")
        f, axs = plt.subplots(1, 2, figsize=(15, 5))
        voxNum = np.random.randint(0, n_vertices)
        axs[0].plot(data_detrended[voxNum, 5:])
        axs[0].set_title(f'Detrended Signal (Vertex {voxNum})')
        axs[0].set_xlabel('TR')
        axs[0].set_ylabel('Amplitude')

        # Plot the power spectrum (squared magnitude)
        axs[1].plot(freqs[1:], np.abs(data_fft[voxNum, 1:])**2, color='deepskyblue')
        axs[1].set_title(f'Power Spectrum (Vertex {voxNum})')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Power (|X(f)|^2)')
        
        plt.tight_layout()
        sample_plot_path = os.path.join(snr_dir, f'signal_fft_sample_{hemi}.png')
        plt.savefig(sample_plot_path, dpi=150)
        plt.close(f)
        print(f"Saved: {sample_plot_path}")

        # 3. Compute relative low frequency power (Rel. Pow) in target band (e.g. 0.005 to 0.08 Hz)
        f_min = 0.005  # lower task frequency bound
        f_max = 0.08   # upper task frequency bound
        print(f"Computing relative low frequency power in band [{f_min}, {f_max}] Hz...")
        
        # Power masks
        low_freq_mask = (freqs >= f_min) & (freqs <= f_max)
        total_freq_mask = freqs > 0  # Discard DC component (0 Hz)
        
        data_fft_power = np.abs(data_fft)**2
        denom = np.sum(data_fft_power[:, total_freq_mask], axis=1)
        denom[denom == 0] = 1.0
        lowFreqPow = np.sum(data_fft_power[:, low_freq_mask], axis=1) / denom

        # 4. Save histogram of low frequency power
        print("Generating relative power histogram...")
        f = plt.figure(figsize=(8, 6))
        plt.hist(lowFreqPow, bins=1000, color='crimson')
        plt.title(f'Histogram of Rel. Low Freq Power ({hemi.capitalize()} Hemisphere)')
        plt.xlabel(f'Relative Low Frequency Power ({f_min} - {f_max} Hz)')
        plt.ylabel('Count')
        hist_plot_path = os.path.join(snr_dir, f'low_freq_power_histogram_{hemi}.png')
        plt.savefig(hist_plot_path, dpi=150)
        plt.close(f)
        print(f"Saved: {hist_plot_path}")

        # 5. Signal profiles across different power quantiles
        print("Generating signals across quantiles plot...")
        n_quantiles = 4
        nVtx = 5
        quantile_edges = np.percentile(lowFreqPow, np.linspace(0, 100, n_quantiles+1))
        
        quantile_indices = []
        for i in range(n_quantiles):
            if i < n_quantiles - 1:
                idx = np.where((lowFreqPow >= quantile_edges[i]) & (lowFreqPow < quantile_edges[i+1]))[0]
            else:
                idx = np.where((lowFreqPow >= 0.5))[0]
            quantile_indices.append(idx)

        fig, axs = plt.subplots(n_quantiles, nVtx, figsize=(5*nVtx, 3*n_quantiles), squeeze=False)
        for q in range(n_quantiles):
            for i in range(nVtx):
                if len(quantile_indices[q]) == 0:
                    continue
                vtx = np.random.choice(quantile_indices[q], 1)[0]
                origThis = (data_orig[vtx, :] - np.mean(data_orig[vtx, :])) / np.std(data_orig[vtx, :])
                axs[q, i].plot(origThis, 'w', label='Orig', alpha=0.5)
                axs[q, i].plot(data_detrended[vtx, :], 'r', label='Detrended')
                axs[q, i].legend()
                axs[q, i].set_title(f'q{q+1}: {vtx} Pow: {lowFreqPow[vtx]:.2f} ')

        plt.tight_layout()
        quantile_plot_path = os.path.join(snr_dir, f'quantiles_signals_{hemi}.png')
        plt.savefig(quantile_plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {quantile_plot_path}")

        # 6. Correlate with model fit estimates if they exist
        # Check fitEstimatesOrig first, then fitEstimates
        paths_to_check = [
            os.path.join(p['popeyeFitDir'], 'fitEstimatesOrig', f'RF_ss5_gFit_popeye_{hemi}.func.gii'),
            os.path.join(p['popeyeFitDir'], 'fitEstimates', f'RF_ss5_gFit_popeye_{hemi}.func.gii')
        ]
        
        gridEstimfPath = None
        for path in paths_to_check:
            if os.path.exists(path):
                gridEstimfPath = path
                break
        
        if gridEstimfPath:
            print(f"Loading grid estimates from {gridEstimfPath}...")
            gridEstimF = nib.load(gridEstimfPath)
            gridEstimData = np.array([x.data for x in gridEstimF.darrays]).T
            
            print("Generating R2 vs Rel. Power correlation plot...")
            f = plt.figure(figsize=(8, 8))
            plt.plot(gridEstimData[:, 1], lowFreqPow, 'ro', markersize=0.25, alpha=0.6)
            plt.plot([0, 1], [0, 1], 'w--', alpha=0.5)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('Model R2')
            plt.ylabel('Rel. Low Freq Power')
            plt.title(f'R2 vs Rel. Low Freq Power ({hemi.capitalize()} Hemisphere)')
            
            corr_plot_path = os.path.join(snr_dir, f'r2_vs_low_freq_power_{hemi}.png')
            plt.savefig(corr_plot_path, dpi=150)
            plt.close(f)
            print(f"Saved: {corr_plot_path}")
        else:
            print(f"No grid fit estimates found for {hemi} hemisphere at checked paths. Skipping correlation plot.")

    print("\nSNR Analysis completed successfully!")

if __name__ == '__main__':
    main()
