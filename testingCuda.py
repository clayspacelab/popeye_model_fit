import numpy as np
import cupy as cp
import time

# Generate synthetic data (observed data)
np.random.seed(0)
x = np.linspace(0, 10, 100)  # independent variable
true_slope = 2.5
true_intercept = -1.0
noise = np.random.normal(0, 1, x.shape)  # Add some noise
y_observed = true_slope * x + true_intercept + noise

# Define a parameter grid for slope and intercept
slopes = np.linspace(2.0, 3.0, 100)  # Grid for slope
intercepts = np.linspace(-2.0, 0.0, 100)  # Grid for intercept
grid_slope, grid_intercept = np.meshgrid(slopes, intercepts)
param_grid = np.array([grid_slope.ravel(), grid_intercept.ravel()]).T

# Function to compute predictions and mean squared error
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Brute-force grid search with NumPy
start_time_np = time.time()
mse_errors_np = []
for slope, intercept in param_grid:
    y_pred = slope * x + intercept
    mse = compute_mse(y_observed, y_pred)
    mse_errors_np.append(mse)
best_mse_np = np.min(mse_errors_np)
end_time_np = time.time()
print(f"NumPy grid search time: {end_time_np - start_time_np:.4f} seconds")
print(f"NumPy best MSE: {best_mse_np:.4f}")

# Brute-force grid search with CuPy
x_cp = cp.asarray(x)
y_observed_cp = cp.asarray(y_observed)
param_grid_cp = cp.asarray(param_grid)

start_time_cp = time.time()
mse_errors_cp = []
for slope, intercept in param_grid_cp:
    y_pred_cp = slope * x_cp + intercept
    mse_cp = cp.mean((y_observed_cp - y_pred_cp) ** 2)
    mse_errors_cp.append(mse_cp)
mse_errors_cp = cp.asnumpy(cp.array(mse_errors_cp))
best_mse_cp = np.min(mse_errors_cp)
end_time_cp = time.time()
print(f"CuPy grid search time: {end_time_cp - start_time_cp:.4f} seconds")
print(f"CuPy best MSE: {best_mse_cp:.4f}")