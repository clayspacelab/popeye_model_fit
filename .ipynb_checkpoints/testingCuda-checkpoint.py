import numpy as np
import cupy as cp
import time

# Generate synthetic data (observed data)
np.random.seed(0)
x = np.linspace(0, 10, 100)  # independent variable

nObs = 100
trueSlopes = np.linspace(1, 10, nObs)
trueIntercepts = np.linspace(1, 10, nObs)

# Generate y_observed_set with varying slopes and intercepts, and added noise
y_observed_set = np.array([
    trueSlope * x + trueIntercept + np.random.normal(0, 1, x.shape)
    for trueSlope, trueIntercept in zip(trueSlopes, trueIntercepts)
])

# Define a parameter grid for slope and intercept
ngridSize = 1000
slopes = np.linspace(2.0, 3.0, ngridSize)  # Grid for slope
intercepts = np.linspace(-2.0, 0.0, ngridSize)  # Grid for intercept
grid_slope, grid_intercept = np.meshgrid(slopes, intercepts)
param_grid = np.array([grid_slope.ravel(), grid_intercept.ravel()]).T

# Function to compute predictions and mean squared error
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def bruteSearch(x, y_observed_set, param_grid, use_gpu=False):
    if use_gpu:
        # Brute-force grid search with CuPy
        x_cp = cp.asarray(x)
        param_grid_cp = cp.asarray(param_grid)
        y_observed_set_cp = cp.asarray(y_observed_set)
        
        start_time_cp = time.time()
        best_mses_cp = []

        # Reshape x to allow broadcasting with param_grid_cp
        x_expanded = x_cp[None, :, None]  # Shape: (1, num_points, 1)
        
        # Extract slopes and intercepts from param_grid
        slopes = param_grid_cp[:, 0][:, None, None]  # Shape: (num_params, 1, 1)
        intercepts = param_grid_cp[:, 1][:, None, None]  # Shape: (num_params, 1, 1)
        
        # Compute predictions for all parameter combinations
        y_pred_all = slopes * x_expanded + intercepts  # Shape: (num_params, num_points, 1)
        
        # Iterate over each observed set, computing the MSE for each param combination
        for y_observed in y_observed_set_cp:
            # Reshape y_observed for broadcasting with y_pred_all
            y_observed_expanded = y_observed[None, :, None]  # Shape: (1, num_points, 1)
        
            # Compute MSE for each parameter combination
            mse_errors_cp = cp.mean((y_observed_expanded - y_pred_all) ** 2, axis=1)  # Shape: (num_params, 1)
        
            # Find the minimum MSE across all parameter combinations
            best_mse_cp = cp.min(mse_errors_cp).item()  # Convert to Python scalar
            best_mses_cp.append(best_mse_cp)
    
        end_time_cp = time.time()
        print(f"CuPy grid search time: {end_time_cp - start_time_cp:.4f} seconds")
        print(f"CuPy best MSE: {best_mse_cp:.4f}")
    else:
        # Brute-force grid search with NumPy
        start_time_np = time.time()
        best_mses_np = []

        for y_observed in y_observed_set:
            mse_errors_np = []
            for slope, intercept in param_grid:
                y_pred = slope * x + intercept
                mse = compute_mse(y_observed, y_pred)
                mse_errors_np.append(mse)
            best_mse_np = np.min(mse_errors_np)
            best_mses_np.append(best_mse_np)

        end_time_np = time.time()
        print(f"NumPy grid search time: {end_time_np - start_time_np:.4f} seconds")
        print(f"NumPy best MSE: {best_mse_np:.4f}")

# Run without GPU
bruteSearch(x, y_observed_set, param_grid, use_gpu=False)
bruteSearch(x, y_observed_set, param_grid, use_gpu=True)