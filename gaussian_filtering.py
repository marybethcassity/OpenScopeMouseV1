import numpy as np
from scipy.optimize import curve_fit

def gaussian_2d(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """2D Gaussian function"""
    x, y = coords
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


def fit_gaussian_to_rf(rf):
    
    # Create coordinate arrays
    x = np.arange(rf.shape[1])
    y = np.arange(rf.shape[0])
    x, y = np.meshgrid(x, y)
    
    # Find the actual maximum location as a better initial guess
    max_idx = np.unravel_index(np.argmax(rf), rf.shape)
    
    # Initial guess for parameters
    amplitude_guess = rf.max() - rf.min()
    xo_guess = max_idx[1]  # Column index (x)
    yo_guess = max_idx[0]  # Row index (y)
    sigma_guess = min(rf.shape) / 4
    
    initial_guess = (amplitude_guess, xo_guess, yo_guess, sigma_guess, sigma_guess, 0, rf.min())
    
    # Set bounds to keep the center within the RF array (with some margin)
    bounds = (
        [0, -1, -1, 0.5, 0.5, -np.pi, -np.inf],  # Lower bounds (allow slight overhang)
        [np.inf, rf.shape[1], rf.shape[0], rf.shape[1]*3, rf.shape[0]*3, np.pi, np.inf]  # Upper bounds
    )
    
    try:
        # Fit the Gaussian with bounds
        popt, pcov = curve_fit(
            gaussian_2d, 
            (x, y), 
            rf.ravel(), 
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )
        
        # Calculate fitted RF
        fitted_rf = gaussian_2d((x, y), *popt).reshape(rf.shape)
        
        # Calculate R-squared
        ss_res = np.sum((rf.ravel() - fitted_rf.ravel())**2)
        ss_tot = np.sum((rf.ravel() - np.mean(rf.ravel()))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Only reject if the fit is extremely poor or center is way out of bounds
        if r_squared < -0.5:  # Very negative R² indicates terrible fit
            print(f"Warning: R² = {r_squared:.3f} is extremely low, rejecting fit")
            return None, 0.0, None
            
        # Check if center is completely outside the reasonable range
        if popt[1] < -2 or popt[1] > rf.shape[1] + 2 or popt[2] < -2 or popt[2] > rf.shape[0] + 2:
            print(f"Warning: Fitted center ({popt[1]:.2f}, {popt[2]:.2f}) is far outside RF bounds")
            return None, 0.0, None
        
        return popt, r_squared, fitted_rf
        
    except RuntimeError as e:
        # If optimization fails, try without bounds
        print(f"Bounded fit failed, trying unbounded fit: {e}")
        try:
            popt, pcov = curve_fit(
                gaussian_2d, 
                (x, y), 
                rf.ravel(), 
                p0=initial_guess,
                maxfev=10000
            )
            
            # Calculate fitted RF
            fitted_rf = gaussian_2d((x, y), *popt).reshape(rf.shape)
            
            # Calculate R-squared
            ss_res = np.sum((rf.ravel() - fitted_rf.ravel())**2)
            ss_tot = np.sum((rf.ravel() - np.mean(rf.ravel()))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
            return popt, r_squared, fitted_rf
            
        except Exception as e2:
            print(f"Unbounded fit also failed: {e2}")
            return None, 0.0, None
            
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, 0.0, None


def filter_rfs_by_gaussian_fit(unit_rfs, r_squared_threshold=0.5, verbose=False):
    """
    Filter receptive fields based on Gaussian fit quality.
    
    Parameters:
    -----------
    unit_rfs : list
        List of 2D receptive field arrays
    r_squared_threshold : float
        Minimum R² value to pass filter
    verbose : bool
        Print filtering statistics
    
    Returns:
    --------
    tuple : (filtered_rfs, filtered_indices, r_squared_values, fitted_rfs)
        - filtered_rfs: List of RFs that passed the filter
        - filtered_indices: Indices of RFs that passed
        - r_squared_values: R² values for all RFs
        - fitted_rfs: List of fitted Gaussian RFs for passing units
    """
    filtered_rfs = []
    filtered_indices = []
    r_squared_values = []
    fitted_rfs = []
    
    for i, rf in enumerate(unit_rfs):
        popt, r_squared, fitted_rf = fit_gaussian_to_rf(rf)
        r_squared_values.append(r_squared)
        
        if r_squared >= r_squared_threshold:
            filtered_rfs.append(rf)
            filtered_indices.append(i)
            fitted_rfs.append(fitted_rf)
    
    if verbose:
        print(f"    Gaussian filtering results:")
        print(f"      Total units: {len(unit_rfs)}")
        print(f"      Passed filter (R² >= {r_squared_threshold}): {len(filtered_rfs)}")
        print(f"      Rejected: {len(unit_rfs) - len(filtered_rfs)}")
        print(f"      Mean R²: {np.mean(r_squared_values):.3f}")
        print(f"      Median R²: {np.median(r_squared_values):.3f}")
        print(f"      Min R²: {np.min(r_squared_values):.3f}")
        print(f"      Max R²: {np.max(r_squared_values):.3f}")
    
    return filtered_rfs, filtered_indices, r_squared_values, fitted_rfs