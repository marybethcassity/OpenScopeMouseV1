"""
Data processing module for NWB files.
Handles loading, unit selection, filtering RFs, and metric calculation.
"""

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import scipy.stats
from scipy.optimize import curve_fit
import warnings

from gaussian_filtering import fit_gaussian_to_rf, get_rf_size_degrees


def deg2rad(arr):
    """Converts array-like input from degrees to radians"""
    return arr / 180 * np.pi

# from lines 812 - 840 of AllenSDK/allensdk/brain_observatory/ecephys/stimulus_analysis/stimulus_analysis.py
# https://github.com/AllenInstitute/AllenSDK/blob/a9b5c685396126d9748f1ccecf7c00f440569f69/allensdk/brain_observatory/ecephys/stimulus_analysis/stimulus_analysis.py
def osi(orivals, tuning):
    """
    Computes the orientation selectivity of a cell using normalized
    circular variance (CirVar) as described in Ringbach 2002.
    
    From AllenSDK stimulus_analysis.py
    
    Parameters
    ----------
    orivals : complex array of length N
         Each value the orientation of the stimulus (in radians).
    tuning : float array of length N
        Each value the (averaged) response of the cell at a different
        orientation.
    
    Returns
    -------
    osi : float
        Circular variance (scalar value, in radians) of the responses.
    """
    if len(orivals) == 0 or len(orivals) != len(tuning):
        warnings.warn('orivals and tunings are of different lengths')
        return np.nan
    
    tuning_sum = tuning.sum()
    if tuning_sum == 0.0:
        return np.nan
    
    cv_top = tuning * np.exp(1j * 2 * orivals)
    return np.abs(cv_top.sum()) / tuning_sum

# from lines 843 - 860 of AllenSDK/allensdk/brain_observatory/ecephys/stimulus_analysis/stimulus_analysis.py
# https://github.com/AllenInstitute/AllenSDK/blob/a9b5c685396126d9748f1ccecf7c00f440569f69/allensdk/brain_observatory/ecephys/stimulus_analysis/stimulus_analysis.py
def dsi(orivals, tuning):
    """
    Computes the direction selectivity of a cell.
    See Ringbach 2002, Van Hooser 2014.
    
    From AllenSDK stimulus_analysis.py
    
    Parameters
    ----------
    orivals : complex array of length N
         Each value the orientation of the stimulus (in radians).
    tuning : float array of length N
        Each value the (averaged) response of the cell at a different
        orientation.
    
    Returns
    -------
    dsi : float
        Direction selectivity index (scalar value, in radians).
    """
    if len(orivals) == 0 or len(orivals) != len(tuning):
        warnings.warn('orivals and tunings are of different lengths')
        return np.nan
    
    tuning_sum = tuning.sum()
    if tuning_sum == 0.0:
        return np.nan
    
    cv_top = tuning * np.exp(1j * orivals)
    return np.abs(cv_top.sum()) / tuning_sum

# from lines 290 - 319 of AllenSDK/allensdk/brain_observatory/ecephys/stimulus_analysis/drifting_gratings.py
# https://github.com/AllenInstitute/AllenSDK/blob/a9b5c685396126d9748f1ccecf7c00f440569f69/allensdk/brain_observatory/ecephys/stimulus_analysis/drifting_gratings.py
def calculate_osi_dsi(unit_id, pref_tf, conditionwise_stats, stimulus_conditions, orivals):
    """
    Calculate OSI and DSI for a unit at their preferred temporal frequency.
    
    From AllenSDK drifting_gratings.py
    
    Parameters
    ----------
    unit_id : int
        Unit ID
    pref_tf : float
        Preferred temporal frequency
    conditionwise_stats : pd.DataFrame
        Conditionwise spike statistics
    stimulus_conditions : pd.DataFrame
        Stimulus conditions table
    orivals : np.array
        Array of orientation values (not used, kept for compatibility)
        
    Returns
    -------
    osi_value : float
        Orientation selectivity index
    dsi_value : float
        Direction selectivity index
    """
    # Get condition indices for this temporal frequency
    condition_inds = stimulus_conditions[stimulus_conditions['temporal_frequency'] == pref_tf].index.values
    
    # Get the spike statistics for this unit and these conditions
    df = conditionwise_stats.loc[unit_id].loc[condition_inds]
    
    # Add orientation column
    df = df.assign(ori=stimulus_conditions.loc[df.index.values]['orientation'])
    
    # Average across repetitions for each orientation
    tuning = df.groupby('ori')['spike_mean'].mean().sort_index().values
    
    # Get unique orientations (sorted)
    unique_oris = np.sort(df['ori'].unique())
    
    # Convert orientations to radians
    orivals_rad = deg2rad(unique_oris).astype('complex128')
    
    # Calculate OSI and DSI
    return osi(orivals_rad, tuning), dsi(orivals_rad, tuning)

def calculate_osi_dsi_all_tf(unit_id, conditionwise_stats, stimulus_conditions):
    """
    Calculate OSI and DSI averaged across all temporal frequencies.
    Used when preferred TF is a continuous float (e.g. from Gaussian fit)
    and cannot be matched to a discrete condition.
    """
    # Get all condition indices
    all_condition_inds = stimulus_conditions.index.values
    
    df = conditionwise_stats.loc[unit_id].loc[all_condition_inds]
    df = df.assign(ori=stimulus_conditions.loc[df.index.values]['orientation'])
    
    # Average across all TFs and repetitions for each orientation
    tuning = df.groupby('ori')['spike_mean'].mean().sort_index().values
    unique_oris = np.sort(df['ori'].unique())
    orivals_rad = deg2rad(unique_oris).astype('complex128')
    
    return osi(orivals_rad, tuning), dsi(orivals_rad, tuning)


def presentationwise_spike_times(nwb, stim_table, stimulus_presentation_ids=None, unit_ids=None):
    """
    Produce a table associating spike times with units and stimulus presentations.
    
    Based on AllenSDK implementation.
    
    Parameters
    ----------
    nwb : NWBFile
        NWB file object
    stim_table : DataFrame
        Stimulus presentation table
    stimulus_presentation_ids : array-like, optional
        Filter to these stimulus presentations
    unit_ids : array-like, optional
        Filter to these units
    
    Returns
    -------
    pandas.DataFrame :
        Index: spike_time (float)
        Columns: stimulus_presentation_id, unit_id, time_since_stimulus_presentation_onset
    """
    if stimulus_presentation_ids is not None:
        stim_table = stim_table.loc[stimulus_presentation_ids]
    
    # Get units
    units_table = nwb.units.to_dataframe()
    if unit_ids is None:
        unit_ids = units_table.index.values
    
    # Create presentation_times array (alternating start/stop times)
    presentation_times = np.zeros([stim_table.shape[0] * 2])
    presentation_times[::2] = np.array(stim_table['start_time'])
    presentation_times[1::2] = np.array(stim_table['stop_time'])
    
    all_presentation_ids = np.array(stim_table.index.values)
    
    presentation_ids = []
    unit_ids_list = []
    spike_times_list = []
    
    for unit_id in unit_ids:
        # Retrieve spike times for this unit
        unit_row_index = units_table.index.get_loc(unit_id)
        data = nwb.units['spike_times'][unit_row_index]
        
        # Find which presentation each spike belongs to using searchsorted
        indices = np.searchsorted(presentation_times, data) - 1
        
        index_valid = indices % 2 == 0
        
        # Get presentation IDs
        presentations = all_presentation_ids[np.floor(indices / 2).astype(int)]
        
        # Sort by presentation
        sorder = np.argsort(presentations)
        presentations = presentations[sorder]
        index_valid = index_valid[sorder]
        data = data[sorder]
        
        # Find boundaries between different presentations
        changes = np.where(np.ediff1d(presentations, to_begin=1, to_end=1))[0]
        
        for ii, jj in zip(changes[:-1], changes[1:]):
            values = data[ii:jj][index_valid[ii:jj]]
            if values.size == 0:
                continue
            unit_ids_list.append(np.zeros([values.size]) + unit_id)
            presentation_ids.append(np.zeros([values.size]) + presentations[ii])
            spike_times_list.append(values)
    
    if not spike_times_list:
        # If there are no spikes, return empty DataFrame
        return pd.DataFrame(columns=[
            'stimulus_presentation_id',
            'unit_id',
            'time_since_stimulus_presentation_onset'])
    
    pres_ids = np.concatenate(presentation_ids).astype(int)
    spike_df = pd.DataFrame({
        'stimulus_presentation_id': pres_ids,
        'unit_id': np.concatenate(unit_ids_list).astype(int)
    }, index=pd.Index(np.concatenate(spike_times_list), name='spike_time'))
    
    # Add time since stimulus presentation onset
    onset_times = stim_table.loc[all_presentation_ids, "start_time"]
    spikes_with_onset = spike_df.join(onset_times, on=["stimulus_presentation_id"])
    spikes_with_onset["time_since_stimulus_presentation_onset"] = (
        spikes_with_onset.index - spikes_with_onset["start_time"]
    )
    spikes_with_onset.sort_values(by='spike_time', axis=0, inplace=True)
    spikes_with_onset.drop(columns=["start_time"], inplace=True)
    
    return spikes_with_onset

# from lines 1491-1500 AllenSDK/allensdk/brain_observatory/ecephys/ecephys_session.py
# https://github.com/AllenInstitute/AllenSDK/blob/a9b5c685396126d9748f1ccecf7c00f440569f69/allensdk/brain_observatory/ecephys/ecephys_session.py
def _extract_summary_count_statistics(index, group):
    """
    Extract summary statistics for spike counts.
    
    Based on AllenSDK implementation.
    """
    return {
        "stimulus_condition_id": index[0],
        "unit_id": index[1],
        "spike_count": group["spike_count"].sum(),
        "stimulus_presentation_count": group.shape[0],
        "spike_mean": np.mean(group["spike_count"].values),
        "spike_std": np.std(group["spike_count"].values, ddof=1),
        "spike_sem": scipy.stats.sem(group["spike_count"].values)
    }

# from lines 891-980 AllenSDK/allensdk/brain_observatory/ecephys/ecephys_session.py
# https://github.com/AllenInstitute/AllenSDK/blob/a9b5c685396126d9748f1ccecf7c00f440569f69/allensdk/brain_observatory/ecephys/ecephys_session.py
def conditionwise_spike_statistics(nwb, stimulus_block='drifting_gratings_field_block_presentations',
                                   stimulus_presentation_ids=None, unit_ids=None):
    """
    Calculate spike statistics grouped by stimulus condition.
    
    Based on AllenSDK implementation.
    
    Parameters
    ----------
    nwb : NWBFile
        NWB file object
    stimulus_block : str
        Which stimulus block to analyze
    stimulus_presentation_ids : array-like, optional
        Filter to these stimulus presentations
    unit_ids : array-like, optional
        Filter to these units
    
    Returns
    -------
    tuple : (result_df, stimulus_conditions)
        result_df: DataFrame with spike statistics indexed by [unit_id, stimulus_condition_id]
        stimulus_conditions: DataFrame with stimulus parameters indexed by stimulus_condition_id
    """
    # Get the appropriate stimulus table
    if stimulus_block not in nwb.intervals:
        available = list(nwb.intervals.keys())
        raise ValueError(f"Stimulus block '{stimulus_block}' not found. Available: {available}")
    
    stim_table = nwb.intervals[stimulus_block].to_dataframe()
    
    if stimulus_presentation_ids is None:
        stimulus_presentation_ids = stim_table.index.values
    
    condition_params = ['orientation', 'temporal_frequency', 'spatial_frequency', 'contrast']
    
    # Get rid of all rows that have NA in condition parameters
    stim_table_clean = stim_table[condition_params + ['start_time', 'stop_time']].dropna(subset=condition_params).copy()
    
    # Convert strings to floats
    for param in condition_params:
        if param in stim_table_clean.columns:
            stim_table_clean[param] = pd.to_numeric(stim_table_clean[param], errors='coerce')
    
    stimulus_presentation_ids = stim_table_clean.index.values
    
    # Create stimulus_condition_id
    stim_table_clean['stimulus_condition_id'] = stim_table_clean.groupby(condition_params).ngroup()
    
    presentations = stim_table_clean.loc[stimulus_presentation_ids].copy()
    
    # Get presentationwise spike times
    spikes = presentationwise_spike_times(nwb, stim_table_clean, stimulus_presentation_ids, unit_ids)
    
    if spikes.empty:
        # No spikes case
        units_table = nwb.units.to_dataframe()
        if unit_ids is None:
            unit_ids = units_table.index.values
        
        spike_counts = pd.DataFrame(
            {'spike_count': 0},
            index=pd.MultiIndex.from_product([
                stimulus_presentation_ids,
                unit_ids],
                names=['stimulus_presentation_id', 'unit_id']))
    else:
        # Count spikes per presentation and unit
        spike_counts = spikes[['stimulus_presentation_id', 'unit_id']].copy()
        spike_counts["spike_count"] = np.zeros(spike_counts.shape[0])
        spike_counts = spike_counts.groupby(["stimulus_presentation_id", "unit_id"]).count()
        
        unit_ids = unit_ids if unit_ids is not None else spikes['unit_id'].unique()
        
        spike_counts = spike_counts.reindex(
            pd.MultiIndex.from_product(
                [stimulus_presentation_ids, unit_ids],
                names=['stimulus_presentation_id', 'unit_id']),
            fill_value=0)
    
    # Merge with presentations to get stimulus_condition_id
    sp = pd.merge(
        spike_counts,
        presentations[['stimulus_condition_id']],
        left_on="stimulus_presentation_id",
        right_index=True,
        how="left"
    )
    sp.reset_index(inplace=True)
    
    # Extract summary statistics
    summary = []
    for ind, gr in sp.groupby(["stimulus_condition_id", "unit_id"]):
        summary.append(_extract_summary_count_statistics(ind, gr))
    
    result_df = pd.DataFrame(summary).set_index(keys=["unit_id", "stimulus_condition_id"])
    
    # Create stimulus_conditions table
    stimulus_conditions = stim_table_clean[condition_params + ['stimulus_condition_id']].drop_duplicates(
        subset='stimulus_condition_id'
    ).set_index('stimulus_condition_id').sort_index()
    
    return result_df, stimulus_conditions


def get_unit_probe(unit_idx, units):
    """Get the probe/device name for a unit."""
    return str(units['device_name'][unit_idx])


def select_condition(unit_idx, units, probe, all_units=False):
    """
    Determine if a unit meets selection criteria.
    
    Parameters:
    -----------
    unit_idx : int
        Index of the unit
    units : DataFrame
        Units table from NWB file
    probe : str
        Target probe name
    all_units : bool
        If True, include all units regardless of SNR
    
    Returns:
    --------
    bool : Whether the unit meets selection criteria
    """
    if all_units:
        return get_unit_probe(unit_idx, units) == probe
    
    return (units["snr"][unit_idx] > 1 and 
            get_unit_probe(unit_idx, units) == probe)


def get_rf(spike_times, xs, ys, rf_stim_table):
    """
    Calculate receptive field from spike times and stimulus presentations.
    
    Parameters:
    -----------
    spike_times : array
        Spike times for the unit
    xs : array
        X positions of RF stimuli
    ys : array
        Y positions of RF stimuli
    rf_stim_table : DataFrame
        Receptive field stimulus presentation table
    
    Returns:
    --------
    ndarray : 2D receptive field response matrix
    """
    unit_rf = np.zeros([ys.size, xs.size])
    
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            # Get stimulus times for this coordinate
            stim_times = rf_stim_table[
                (rf_stim_table.x_position == str(x)) & 
                (rf_stim_table.y_position == str(y))
            ].start_time
            
            response_spike_count = 0
            for stim_time in stim_times:
                # Count spikes within 0.2 seconds after stimulus
                start_idx, end_idx = np.searchsorted(
                    spike_times, 
                    [stim_time, stim_time + 0.2]
                )
                response_spike_count += end_idx - start_idx
            
            unit_rf[yi, xi] = response_spike_count
    
    return unit_rf

# moved this functionality to main compute_pref_variables.py

# def load_nwb_data(nwb_path):
#     """
#     Load NWB file and extract necessary data.
    
#     Parameters:
#     -----------
#     nwb_path : Path
#         Path to NWB file
    
#     Returns:
#     --------
#     dict : Dictionary containing NWB data objects
#     """
#     io = NWBHDF5IO(str(nwb_path), 'r', load_namespaces=True)
#     nwb = io.read()
    
#     return {
#         'nwb': nwb,
#         'io': io,
#         'units': nwb.units.to_dataframe(),
#         'rf_stim_table': nwb.intervals["receptive_field_block_presentations"].to_dataframe(),
#         'dg_stim_table': nwb.intervals["drifting_gratings_field_block_presentations"].to_dataframe()
#     }


def process_units(nwb_data, probe, all_units=False, verbose=False):
    """
    Process units from NWB data to get receptive fields.
    
    Parameters:
    -----------
    nwb_data : dict
        Dictionary from load_nwb_data
    probe : str
        Target probe name
    all_units : bool
        Whether to include all units
    verbose : bool
        Print detailed information
    
    Returns:
    --------
    dict : Dictionary containing unit data
    """
    units = nwb_data['units']
    rf_stim_table = nwb_data['rf_stim_table']
    nwb = nwb_data['nwb']
    
    # Get unique RF positions
    xs = list(set(rf_stim_table.x_position))
    ys = list(set(rf_stim_table.y_position))

    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]

    xs = np.sort(xs)
    ys = np.sort(ys)
    if verbose:
        print(f"    RF grid: xs={xs}, ys={ys}")
    
    # Find units matching criteria
    unit_indices = []
    unit_rfs = []
    
    for unit_idx in range(len(units)):
        if select_condition(unit_idx, units, probe):
            unit_indices.append(unit_idx)
            
            # Get spike times and calculate RF
            spike_times = nwb.units['spike_times'][unit_idx]
            unit_rf = get_rf(spike_times, xs, ys, rf_stim_table)
            unit_rfs.append(unit_rf)
    
    if verbose:
        print(f"    Found {len(unit_indices)} units on {probe}")
        print(f"    RF grid size: {len(xs)} x {len(ys)}")
    
    return {
        'unit_indices': unit_indices,
        'unit_rfs': unit_rfs,
        'xs': xs,
        'ys': ys
    }


def calculate_rf_center(rf, xs, ys):
    """
    Calculate the center of a receptive field using Gaussian fitting.
    
    Parameters:
    -----------
    rf : ndarray
        2D receptive field array
    xs : array
        X positions
    ys : array
        Y positions
    
    Returns:
    --------
    tuple : (x_center, y_center, r_squared) or (None, None, None) if fitting fails
    """
    from gaussian_filtering import fit_gaussian_to_rf
    
    # Fit Gaussian to RF
    popt, r_squared, _ = fit_gaussian_to_rf(rf)
    
    if popt is None:
        # If Gaussian fit failed, return None
        return None, None, None
    
    # Extract center coordinates (xo, yo are at indices 1 and 2)
    x_idx = popt[1]  # Index in the RF array
    y_idx = popt[2]  # Index in the RF array
    
    # Convert xs and ys to float arrays
    xs_float = xs.astype(float)
    ys_float = ys.astype(float)
    
    # Convert from array indices to actual RF positions
    x_pos = np.interp(x_idx, np.arange(len(xs_float)), xs_float)
    y_pos = np.interp(y_idx, np.arange(len(ys_float)), ys_float)
    
    return float(x_pos), float(y_pos), float(r_squared)


def gaussian_2d_sftf(coords, amplitude, log_sf0, log_tf0, sigma_sf, sigma_tf, offset):
    """
    2D Gaussian in log2(SF) x log2(TF) space (no rotation term).
    Axis-aligned, matching the separable assumption for tuning curves.

    Parameters
    ----------
    coords : tuple of (log_sf_grid, log_tf_grid), each flattened
    amplitude : peak amplitude above baseline
    log_sf0 : preferred SF in log2 units
    log_tf0 : preferred TF in log2 units
    sigma_sf : tuning width in log2 SF
    sigma_tf : tuning width in log2 TF
    offset : baseline firing rate
    """
    log_sf, log_tf = coords
    g = offset + amplitude * np.exp(
        -(
            (log_sf - log_sf0) ** 2 / (2 * sigma_sf ** 2) +
            (log_tf - log_tf0) ** 2 / (2 * sigma_tf ** 2)
        )
    )
    return g.ravel()


def fit_sftf_gaussian(sf_vals, tf_vals, response_matrix):
    """
    Fit a 2D Gaussian to an SF x TF response matrix in log2 frequency space.

    Parameters
    ----------
    sf_vals : array-like of shape (n_sf,)
        Spatial frequency values tested (linear units, e.g. cycles/deg)
    tf_vals : array-like of shape (n_tf,)
        Temporal frequency values tested (linear units, e.g. Hz)
    response_matrix : array-like of shape (n_sf, n_tf)
        Mean spike rates; rows index SF, columns index TF

    Returns
    -------
    pref_sf : float or None
        Preferred spatial frequency in linear units
    pref_tf : float or None
        Preferred temporal frequency in linear units
    r_squared : float or None
        Goodness of fit (R²)
    popt : array or None
        Raw fitted parameters [amplitude, log_sf0, log_tf0, sigma_sf, sigma_tf, offset]
    """
    sf_vals = np.array(sf_vals, dtype=float)
    tf_vals = np.array(tf_vals, dtype=float)
    response_matrix = np.array(response_matrix, dtype=float)

    # Build log2 grids
    log_sf = np.log2(sf_vals)
    log_tf = np.log2(tf_vals)
    log_sf_grid, log_tf_grid = np.meshgrid(log_sf, log_tf, indexing='ij')  # shape (n_sf, n_tf)
    z = response_matrix.ravel()

    # Initial guesses from argmax
    flat_idx = np.argmax(response_matrix)
    sf_idx, tf_idx = np.unravel_index(flat_idx, response_matrix.shape)

    amplitude_guess = response_matrix.max() - response_matrix.min()
    log_sf0_guess = log_sf[sf_idx]
    log_tf0_guess = log_tf[tf_idx]
    sigma_sf_guess = (log_sf[-1] - log_sf[0]) / 4.0 if len(log_sf) > 1 else 1.0
    sigma_tf_guess = (log_tf[-1] - log_tf[0]) / 4.0 if len(log_tf) > 1 else 1.0
    offset_guess = response_matrix.min()

    initial_guess = (amplitude_guess, log_sf0_guess, log_tf0_guess, sigma_sf_guess, sigma_tf_guess, offset_guess)

    bounds = (
        [0,          log_sf[0],  log_tf[0],  0.1, 0.1, -np.inf],
        [np.inf,     log_sf[-1], log_tf[-1], 10,  10,   np.inf]
    )

    coords = (log_sf_grid.ravel(), log_tf_grid.ravel())

    def _r_squared(z_data, z_fit):
        ss_res = np.sum((z_data - z_fit) ** 2)
        ss_tot = np.sum((z_data - np.mean(z_data)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    try:
        popt, _ = curve_fit(
            gaussian_2d_sftf,
            coords,
            z,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )
        z_fit = gaussian_2d_sftf(coords, *popt)
        r_squared = _r_squared(z, z_fit)

        if r_squared < -0.5:
            print(f"Warning: SF/TF Gaussian fit R² = {r_squared:.3f}, rejecting")
            return None, None, None, None

        pref_sf = 2 ** popt[1]  # convert log2 back to linear
        pref_tf = 2 ** popt[2]

        return pref_sf, pref_tf, r_squared, popt

    except RuntimeError:
        # Bounded fit failed — retry without bounds
        try:
            popt, _ = curve_fit(
                gaussian_2d_sftf,
                coords,
                z,
                p0=initial_guess,
                maxfev=10000
            )
            z_fit = gaussian_2d_sftf(coords, *popt)
            r_squared = _r_squared(z, z_fit)
            pref_sf = 2 ** popt[1]
            pref_tf = 2 ** popt[2]
            return pref_sf, pref_tf, r_squared, popt

        except Exception as e:
            print(f"SF/TF Gaussian fit failed (unbounded): {e}")
            return None, None, None, None

    except Exception as e:
        print(f"SF/TF Gaussian fit failed: {e}")
        return None, None, None, None

def bin_to_nearest(value, presented_values):
    """
    Bin a continuous fitted value to the nearest presented stimulus value.
    
    Parameters
    ----------
    value : float or None/NaN
        Fitted preferred value (e.g. pref_sf_gaussian)
    presented_values : array-like
        The discrete stimulus values that were actually presented
    
    Returns
    -------
    float : nearest presented stimulus value, or NaN if input is NaN
    """
    if value is None or np.isnan(value):
        return np.nan
    presented_values = np.array(presented_values)
    return presented_values[np.argmin(np.abs(np.log2(presented_values) - np.log2(value)))]

def calculate_all_metrics(nwb_data, units_data, mouse_name, probe, use_curvefit=False, verbose=False):
    """
    Calculate all metrics for all units.
    
    Parameters:
    -----------
    nwb_data : dict
        NWB data dictionary
    units_data : dict
        Processed units data
    mouse_name : str
        Name of the mouse
    probe : str
        Probe name
    use_curvefit : bool
        Whether to use curve fitting for preferred metrics
    verbose : bool
        Print progress
    
    Returns:
    --------
    DataFrame : Results for all units
    """
    results = []
    
    nwb = nwb_data['nwb']
    units = nwb_data['units']
    dg_stim_table = nwb_data['dg_stim_table']
    xs = units_data['xs']
    ys = units_data['ys']
    
    unit_indices = units_data['unit_indices']
    total_units = len(unit_indices)
    
    if total_units == 0:
        return None
    
    print(f"    Processing {total_units} units...")
    

    if verbose:
        print(f"    Calculating spike statistics for all {total_units} units...")
    
    try:
        conditionwise_stats, stimulus_conditions = conditionwise_spike_statistics(
            nwb,
            stimulus_block='drifting_gratings_field_block_presentations',
            stimulus_presentation_ids=dg_stim_table.index.values,
            unit_ids=unit_indices  # Pass all unit IDs at once
        )
    except Exception as e:
        print(f"    Error calculating conditionwise statistics: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None
    
    # Get unique stimulus values
    ori_vals = np.sort(stimulus_conditions['orientation'].unique())
    tf_vals = np.sort(stimulus_conditions['temporal_frequency'].unique())
    sf_vals = np.sort(stimulus_conditions['spatial_frequency'].unique())
    
    if verbose:
        print(f"    Processing individual unit metrics...")
    
    # Process each unit
    for idx, unit_idx in enumerate(unit_indices):
        if verbose and (idx % 50 == 0 or idx == total_units - 1):
            print(f"      Unit {idx+1}/{total_units}...")
        
        try:
            # Get this unit's statistics from the pre-calculated results
            try:
                unit_stats = conditionwise_stats.loc[unit_idx]
            except KeyError:
                if verbose:
                    print(f"      Warning: No statistics found for unit {unit_idx}")
                continue
            
            # Step 1: Calculate preferred orientation (across all TFs and SFs)
            ori_responses = []
            for ori in ori_vals:
                ori_conditions = stimulus_conditions[
                    stimulus_conditions['orientation'] == ori
                ].index.values
                
                try:
                    mean_response = unit_stats.loc[ori_conditions]['spike_mean'].mean()
                    mean_response = float(mean_response)
                except (KeyError, ValueError, TypeError):
                    mean_response = 0.0
                ori_responses.append(mean_response)
            
            pref_ori = ori_vals[np.argmax(ori_responses)]

            # Step 2: Calculate preferred spatial frequency
            sf_responses = []
            for sf in sf_vals:
                ori_sf_conditions = stimulus_conditions[
                    (stimulus_conditions['spatial_frequency'] == sf)
                ].index.values
                
                try:
                    mean_response = unit_stats.loc[ori_sf_conditions]['spike_mean'].mean()
                    mean_response = float(mean_response)
                except (KeyError, ValueError, TypeError):
                    mean_response = 0.0
                sf_responses.append(mean_response)
            
            pref_sf = sf_vals[np.argmax(sf_responses)]
            
            # Step 3: Calculate preferred temporal frequency (at preferred orientation) for nested preferred spatial frequency
            tf_responses = []
            for tf in tf_vals:
                ori_tf_conditions = stimulus_conditions[
                    (stimulus_conditions['orientation'] == pref_ori) & 
                    (stimulus_conditions['temporal_frequency'] == tf)
                ].index.values
                
                try:
                    mean_response = unit_stats.loc[ori_tf_conditions]['spike_mean'].mean()
                    mean_response = float(mean_response)
                except (KeyError, ValueError, TypeError):
                    mean_response = 0.0
                tf_responses.append(mean_response)
            
            pref_tf_temp = tf_vals[np.argmax(tf_responses)]

            # Step 4: Calculate preferred spatial frequency nested (at preferred orientation & preferred TF)
            sf_responses = []
            for sf in sf_vals:
                ori_sf_conditions = stimulus_conditions[
                    (stimulus_conditions['orientation'] == pref_ori) &
                    (stimulus_conditions['temporal_frequency'] == pref_tf_temp) &
                    (stimulus_conditions['spatial_frequency'] == sf)
                ].index.values
                
                try:
                    mean_response = unit_stats.loc[ori_sf_conditions]['spike_mean'].mean()
                    mean_response = float(mean_response)
                except (KeyError, ValueError, TypeError):
                    mean_response = 0.0
                sf_responses.append(mean_response)
            
            pref_sf_nested = sf_vals[np.argmax(sf_responses)]
                                     
            # Step 5: Calculate preferred temporal frequency 
            tf_responses = []
            for tf in tf_vals:
                ori_tf_conditions = stimulus_conditions[
                    (stimulus_conditions['temporal_frequency'] == tf)
                ].index.values
                
                try:
                    mean_response = unit_stats.loc[ori_tf_conditions]['spike_mean'].mean()
                    mean_response = float(mean_response)
                except (KeyError, ValueError, TypeError):
                    mean_response = 0.0
                tf_responses.append(mean_response)
            
            pref_tf = tf_vals[np.argmax(tf_responses)]

            # Step 6: Calculate preferred spatial frequency (at preferred orientation) for nested preferred temporal frequency
            sf_responses = []
            for sf in sf_vals:
                ori_sf_conditions = stimulus_conditions[
                    (stimulus_conditions['orientation'] == pref_ori) & 
                    (stimulus_conditions['spatial_frequency'] == sf)
                ].index.values
                
                try:
                    mean_response = unit_stats.loc[ori_sf_conditions]['spike_mean'].mean()
                    mean_response = float(mean_response)
                except (KeyError, ValueError, TypeError):
                    mean_response = 0.0
                sf_responses.append(mean_response)
            
            pref_sf_temp = sf_vals[np.argmax(sf_responses)]
                                     
            # Step 7: Calculate preferred temporal frequency nested (at preferred orientation & preferred SF)
            tf_responses = []
            for tf in tf_vals:
                ori_tf_conditions = stimulus_conditions[
                    (stimulus_conditions['orientation'] == pref_ori) &
                    (stimulus_conditions['spatial_frequency'] == pref_sf_temp) &
                    (stimulus_conditions['temporal_frequency'] == tf)
                ].index.values
                
                try:
                    mean_response = unit_stats.loc[ori_tf_conditions]['spike_mean'].mean()
                    mean_response = float(mean_response)
                except (KeyError, ValueError, TypeError):
                    mean_response = 0.0
                tf_responses.append(mean_response)
            
            pref_tf_nested = tf_vals[np.argmax(tf_responses)]


            # Step 8: Fit Gaussian to SF x TF responses at preferred orientation to get preferred SF and TF
            pref_sf_gaussian = None
            pref_tf_gaussian = None
            r_squared_gaussian = None

            try:
                sftf_matrix = np.full((len(sf_vals), len(tf_vals)), np.nan)
                for i, sf in enumerate(sf_vals):
                    for j, tf in enumerate(tf_vals):
                        conditions = stimulus_conditions[
                            (stimulus_conditions['orientation'] == pref_ori) &
                            (stimulus_conditions['spatial_frequency'] == sf) &
                            (stimulus_conditions['temporal_frequency'] == tf)
                        ].index.values
                        
                        try:
                            mean_response = unit_stats.loc[conditions]['spike_mean'].mean()
                            mean_response = float(mean_response)
                        except (KeyError, ValueError, TypeError):
                            mean_response = 0.0
                        sftf_matrix[i, j] = mean_response

                sftf_matrix = np.nan_to_num(sftf_matrix, nan=0.0)
                pref_sf_gaussian, pref_tf_gaussian, r_squared_gaussian, popt = fit_sftf_gaussian(sf_vals, tf_vals, sftf_matrix)

                pref_sf_gaussian_binned = bin_to_nearest(pref_sf_gaussian, sf_vals)
                pref_tf_gaussian_binned = bin_to_nearest(pref_tf_gaussian, tf_vals)

            except Exception as e:
                if verbose:
                    print(f"      Warning: Gaussian fitting failed for unit {unit_idx}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Calculate OSI and DSI
            osi_val, dsi_val = calculate_osi_dsi(
                unit_idx, pref_tf, conditionwise_stats, stimulus_conditions, ori_vals
            )

            osi_val_nested, dsi_val_nested = calculate_osi_dsi(
                unit_idx, pref_tf_nested, conditionwise_stats, stimulus_conditions, ori_vals
            )

            if pref_tf_gaussian is not None:
                osi_val_gaussian, dsi_val_gaussian = calculate_osi_dsi_all_tf(
                    unit_idx, conditionwise_stats, stimulus_conditions
                )
            else:
                osi_val_gaussian, dsi_val_gaussian = np.nan, np.nan
            
            peak_dff_dg = float(max(ori_responses)) if ori_responses else 0.0
            
            # Calculate RF center using Gaussian fitting
            rf = units_data['unit_rfs'][idx]
            popt, r_squared, _ = fit_gaussian_to_rf(rf)
            
            if popt is not None:
                x_idx = popt[1]
                y_idx = popt[2]
                x_pos = np.interp(x_idx, np.arange(len(xs)), xs)
                y_pos = np.interp(y_idx, np.arange(len(ys)), ys)

                size_metrics = get_rf_size_degrees(popt, rf.shape)
                rf_fwhm = size_metrics['fwhm_degrees']
                rf_size_deg = size_metrics['size_degrees']
                rf_sigma_x_deg = size_metrics['sigma_x_degrees']
                rf_sigma_y_deg = size_metrics['sigma_y_degrees']
                
            else:
                x_pos = None
                y_pos = None

                rf_fwh = None
                rf_size_deg = None
                rf_sigma_x_deg = None
                rf_sigma_y_deg = None

            
            # Get SNR
            snr = float(units['snr'][unit_idx])
            
            # Compile results
            result = {
                'mouse_name': mouse_name,
                'probe': probe,
                'unit_id': int(unit_idx),
                'rf': rf,
                'pref_ori': float(pref_ori),
                'ori_responses': ori_responses,
                'pref_tf': float(pref_tf),
                'pref_tf_nested': float(pref_tf_nested),
                'pref_tf_gaussian': float(pref_tf_gaussian) if pref_tf_gaussian is not None else np.nan,
                'pref_tf_gaussian_snapped': pref_tf_gaussian_binned,
                'tf_responses': tf_responses,
                'pref_sf': float(pref_sf),
                'pref_sf_nested': float(pref_sf_nested),
                'pref_sf_gaussian': float(pref_sf_gaussian) if pref_sf_gaussian is not None else np.nan,
                'pref_sf_gaussian_snapped': pref_sf_gaussian_binned,
                'sf_responses': sf_responses,
                'osi_dg': float(osi_val),
                'osi_dg_nested': float(osi_val_nested),
                'osi_dg_gaussian': float(osi_val_gaussian) if osi_val_gaussian is not None else np.nan,
                'dsi_dg': float(dsi_val),
                'dsi_dg_nested': float(dsi_val_nested),
                'dsi_dg_gaussian': float(dsi_val_gaussian) if dsi_val_gaussian is not None else np.nan,
                'peak_dff_dg': peak_dff_dg,
                'rf_x_center': x_pos if x_pos is not None else np.nan,
                'rf_y_center': y_pos if y_pos is not None else np.nan,
                'rf_r_squared': r_squared if r_squared is not None else np.nan,
                'r_squared_gaussian': r_squared_gaussian if r_squared_gaussian is not None else np.nan,
                'snr': snr,
                'rf_fwhm_deg': rf_fwhm if rf_fwhm is not None else np.nan,
                'rf_size_deg': rf_size_deg if rf_size_deg is not None else np.nan,
                'rf_sigma_x_deg': rf_sigma_x_deg if rf_sigma_x_deg is not None else np.nan,
                'rf_sigma_y_deg': rf_sigma_y_deg if rf_sigma_y_deg is not None else np.nan
            }
            
            # Add R² from filtering if available
            if 'r2_values' in units_data and idx < len(units_data['r2_values']):
                result['r_squared_filter'] = units_data['r2_values'][idx]
            
            results.append(result)
            
        except Exception as e:
            if verbose:
                print(f"      Warning: Error processing unit {unit_idx}: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    if use_curvefit:
        print("    Warning: Curve fitting not yet implemented, using argmax")
    
    return pd.DataFrame(results) if results else None