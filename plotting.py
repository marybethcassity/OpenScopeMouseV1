"""
Plotting module for visualizing analysis results.
Includes summary figures, distributions, and receptive field examples.
Functions extracted from preferred_metrics_new.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import curve_fit
from matplotlib.colors import ListedColormap
import pandas as pd


def plot_metric_distributions(df, units_data, output_dir, probe_name=None):
    """
    Create distribution plots for OSI, DSI, and preferred metrics.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe with columns: mouse_name, osi, dsi, pref_ori, pref_tf, pref_sf
    units_data : dict
        Units data, expected to contain 'filtered_rfs' key if RF plots are desired
    output_dir : Path
        Output directory
    probe_name : str, optional
        Probe name for labeling
    """
    output_dir = Path(output_dir)
    mouse_name = df['mouse_name'].unique()[0] if df['mouse_name'].nunique() == 1 else None

    shared_kwargs = dict(
        peak_dff_min=1.0,
        output_dir=output_dir,
        probe_name=probe_name,
        mouse_name=mouse_name,
    )

    # OSI distribution (non-nested and nested)
    for variant_kwargs in [{"nested": False}, {"nested": True}]:
        plot_orientation_selectivity(df, n_hist_bins=20, **shared_kwargs, **variant_kwargs)

    # DSI distribution (no nested version)
    plot_direction_selectivity(
        df,
        n_hist_bins=20,
        peak_dff_min=1.0,
        save_path=output_dir / 'dsi_distribution.png',
        probe_name=probe_name,
        mouse_name=mouse_name
    )

    # Preferred orientation bar (no nested version)
    plot_preferred_orientation_bar(df, **shared_kwargs)

    # Preferred TF and SF bar charts (non-nested and nested)
    bar_configs = [
        (plot_preferred_tf_bar, [{"nested": False}, {"nested": True}]),
        (plot_preferred_sf_bar, [{"nested": False}, {"nested": True}]),
    ]

    for plot_fn, variants in bar_configs:
        for variant_kwargs in variants:
            plot_fn(df, **shared_kwargs, **variant_kwargs)

    # RF center plots (only if RF data is available)
    if 'rf_x_center' in df.columns and 'rf_y_center' in df.columns:
        filtered_rfs = units_data.get('filtered_rfs', None)

        rf_shared_kwargs = dict(
            output_dir=output_dir,
            probe_name=probe_name,
            mouse_name=mouse_name,
            background=False,
        )

        rf_configs = [
            (plot_preferred_orientation_by_rf_from_csv, [
                {"binned": False},
                {"binned": True},
            ]),
            (plot_osi_by_rf_from_csv, [
                {"nested": False},
                {"nested": True},
                {"nested": False, "binned": True},
                {"nested": True,  "binned": True},
            ]),
            (plot_preferred_tf_by_rf_from_csv, [
                {"nested": False},
                {"nested": True},
                {"nested": False, "binned": True},
                {"nested": True,  "binned": True},
            ]),
            (plot_preferred_sf_by_rf_from_csv, [
                {"nested": False},
                {"nested": True},
                {"nested": False, "binned": True},
                {"nested": True,  "binned": True},
            ]),
        ]

        for plot_fn, variants in rf_configs:
            for variant_kwargs in variants:
                plot_fn(df, filtered_rfs, **rf_shared_kwargs, **variant_kwargs)

def plot_summary_figures(df, units_data, output_dir, probe_name=None):
    """
    Create summary scatter plots of preferred metrics vs RF position.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    units_data : dict
        Dictionary containing 'unit_rfs' or can be None
    output_dir : Path
        Output directory
    probe_name : str, optional
        Probe name for labeling
    """
    output_dir = Path(output_dir)
    
    # Get mouse name if only one mouse
    mouse_name = df['mouse_name'].unique()[0] if df['mouse_name'].nunique() == 1 else None
    
    # Extract RFs from the DataFrame 'rf' column
    filtered_rfs = None
    if 'rf' in df.columns:
        # Extract RF arrays from DataFrame
        filtered_rfs = [rf for rf in df['rf'].values if rf is not None]
        if len(filtered_rfs) == 0:
            filtered_rfs = None
    
    # If DataFrame doesn't have RFs, try units_data
    if filtered_rfs is None and units_data is not None:
        if 'unit_rfs' in units_data:
            filtered_rfs = units_data['unit_rfs']
    
    # Plot average RF (if RFs are available)
    if filtered_rfs is not None:
        plot_avg_rf(
            df,
            filtered_rfs,
            save_path=output_dir / 'average_rf.png',
            probe_name=probe_name,
            mouse_name=mouse_name
        )
    
    # Check if we have RF center data
    if 'rf_x_center' in df.columns and 'rf_y_center' in df.columns:
        # Define all plot functions and their variant configurations
        plot_configs = [
            # (function, variants) - variants are lists of kwargs to pass
            (plot_preferred_orientation_by_rf_from_csv, [
                {"binned": True},
                {"binned": False},
            ]),
            (plot_osi_by_rf_from_csv, [
                {"nested": False},
                {"nested": True},
                {"nested": False, "binned": True},
                {"nested": True,  "binned": True},
            ]),
            (plot_preferred_tf_by_rf_from_csv, [
                {"nested": False},
                {"nested": True},
                {"nested": False, "binned": True},
                {"nested": True,  "binned": True},
            ]),
            (plot_preferred_sf_by_rf_from_csv, [
                {"nested": False},
                {"nested": True},
                {"nested": False, "binned": True},
                {"nested": True,  "binned": True},
            ]),
        ]

        # Shared kwargs for all plots
        shared_kwargs = dict(
            output_dir=output_dir,
            probe_name=probe_name,
            mouse_name=mouse_name,
            background=False,
        )

        # Call all combinations
        for plot_fn, variants in plot_configs:
            for variant_kwargs in variants:
                plot_fn(df, filtered_rfs, **shared_kwargs, **variant_kwargs)
        
        # Plot RF position vs preferred orientation
        plot_rf_position_vs_pref_ori_from_csv(
            df,
            save_path_x=output_dir / 'rf_x_position_vs_pref_ori.png',
            save_path_y=output_dir / 'rf_y_position_vs_pref_ori.png',
            probe_name=probe_name,
            mouse_name=mouse_name
        )
        
        # Plot RF position vs preferred temporal frequency
        plot_rf_position_vs_pref_tf_from_csv(
            df,
            save_path_x=output_dir / 'rf_x_position_vs_pref_tf.png',
            save_path_y=output_dir / 'rf_y_position_vs_pref_tf.png',
            probe_name=probe_name,
            mouse_name=mouse_name
        )
        
        # Plot RF position vs preferred spatial frequency
        plot_rf_position_vs_pref_sf_from_csv(
            df,
            save_path_x=output_dir / 'rf_x_position_vs_pref_sf.png',
            save_path_y=output_dir / 'rf_y_position_vs_pref_sf.png',
            probe_name=probe_name,
            mouse_name=mouse_name
        )
# ============================================================================
# Core plotting functions from preferred_metrics_new.ipynb
# ============================================================================

def plot_avg_rf(df, 
                filtered_rfs, 
                save_path=None, 
                probe_name=None, 
                mouse_name=None):
    """
    Plot the average receptive field.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe with RF center positions
    filtered_rfs : list or array
        List of receptive field arrays (2D) for each unit
    save_path : str or Path, optional
        Path to save figure
    probe_name : str, optional
        Probe name for title
    mouse_name : str, optional
        Mouse name for title
    """
    # Filter out rows without RF center data
    df = df.dropna(subset=['rf_x_center', 'rf_y_center'])
    
    if len(df) == 0:
        print("No valid RF center data found")
        return
    
    if filtered_rfs is None or len(filtered_rfs) == 0:
        print("No RFs available to plot")
        return
    
    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate and plot average RF
    average_rf = np.mean(filtered_rfs, axis=0)
    
    # Get RF extent from positions
    x_min, x_max = -40, 40
    y_min, y_max = -40, 40
    
    # Add padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    im = ax.imshow(average_rf, origin="lower", cmap='viridis', alpha=0.8, 
                   extent=[x_min - x_padding, x_max + x_padding, 
                           y_min - y_padding, y_max + y_padding],
                   aspect='auto')
    
    plt.colorbar(im, ax=ax, label='Average Response')
    
    ax.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax.set_ylabel('RF Center Y Position (degrees)', fontsize=14)
    
    # Create title
    title = f'Average Receptive Field\n({len(filtered_rfs)} cells)'
    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n' + title
    elif probe_name:
        title = f'{probe_name}\n' + title
    elif mouse_name:
        title = f'{mouse_name}\n' + title
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_orientation_selectivity(peak_df,
                                 si_range=(0, 1),
                                 n_hist_bins=20,
                                 peak_dff_min=1.0,
                                 output_dir=None,
                                 density=True,
                                 probe_name=None,
                                 mouse_name=None,
                                 nested=False):
    """
    Plot orientation selectivity histogram.

    Parameters
    ----------
    nested : bool, default=False
        If True, use nested OSI values
    """
    vis_cells = (peak_df.peak_dff_dg > peak_dff_min)

    if nested:
        osi_cells = vis_cells & (peak_df.osi_dg_nested > si_range[0]) & (peak_df.osi_dg_nested < si_range[1])
        osis = peak_df.loc[osi_cells].osi_dg_nested.values
        title = f'Nested Orientation Selectivity Distribution\n({len(osis)} cells)'
        save_path = output_dir / 'osi_distribution_nested.png' if output_dir else None
    else:
        osi_cells = vis_cells & (peak_df.osi_dg > si_range[0]) & (peak_df.osi_dg < si_range[1])
        osis = peak_df.loc[osi_cells].osi_dg.values
        title = f'Orientation Selectivity Distribution\n({len(osis)} cells)'
        save_path = output_dir / 'osi_distribution.png' if output_dir else None

    if len(osis) == 0:
        print("No cells passed OSI filter")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(osis, bins=n_hist_bins, edgecolor='black', alpha=0.7,
            cumulative=False, density=density, color='steelblue')
    ax.set_xlabel('Orientation Selectivity Index (OSI)', fontsize=12)
    ylabel = 'Number of Cells (Normalized)' if density else 'Number of Cells'
    ax.set_ylabel(ylabel, fontsize=12)

    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n' + title
    elif probe_name:
        title = f'{probe_name}\n' + title
    elif mouse_name:
        title = f'{mouse_name}\n' + title

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    stats_text = f'Mean: {np.mean(osis):.3f}\nMedian: {np.median(osis):.3f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_direction_selectivity(peak_df, 
                               si_range=(0, 1),
                               n_hist_bins=20,
                               peak_dff_min=1.0,
                               save_path=None,
                               density=True,
                               probe_name=None,
                               mouse_name=None):
    """
    Plot direction selectivity index (DSI) histogram.
    """
    # Filter responsive cells
    vis_cells = (peak_df.peak_dff_dg > peak_dff_min)
    
    # Filter direction selective cells
    dsi_cells = vis_cells & (peak_df.dsi_dg > si_range[0]) & (peak_df.dsi_dg < si_range[1])
    
    peak_dsi = peak_df.loc[dsi_cells]
    dsis = peak_dsi.dsi_dg.values
    
    if len(dsis) == 0:
        print("No cells passed DSI filter")
        return
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(dsis, bins=n_hist_bins, edgecolor='black', alpha=0.7, 
            cumulative=False, density=density, color='green')
    ax.set_xlabel('Direction Selectivity Index (DSI)', fontsize=12)
    ylabel = 'Number of Cells (Normalized)' if density else 'Number of Cells'
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Create title with probe and mouse info
    title = f'Direction Selectivity Distribution\n({len(dsis)} cells)'
    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n' + title
    elif probe_name:
        title = f'{probe_name}\n' + title
    elif mouse_name:
        title = f'{mouse_name}\n' + title
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add stats text
    stats_text = f'Mean: {np.mean(dsis):.3f}\nMedian: {np.median(dsis):.3f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #print(f"✓ Saved DSI figure to: {save_path}")
    
    plt.close()


def plot_preferred_orientation_bar(peak_df,
                                   peak_dff_min=1.0,
                                   output_dir=None,
                                   color='mediumorchid',
                                   probe_name=None,
                                   mouse_name=None,
                                   normalize=False):
    """
    Plot preferred orientation as a bar histogram.

    Parameters
    ----------
    normalize : bool, default=False
        If True, normalize counts to percentages (0-100)
    """
    vis_cells = (peak_df.peak_dff_dg > peak_dff_min)
    pref_oris = peak_df.loc[vis_cells].pref_ori.dropna().values

    if len(pref_oris) == 0:
        print("No cells with preferred orientation data")
        return

    title = f'Preferred Orientation Distribution\n({len(pref_oris)} cells)'
    save_path = output_dir / 'pref_ori_distribution.png' if output_dir else None

    unique_oris = np.sort(np.unique(pref_oris))
    counts = np.array([np.sum(pref_oris == ori) for ori in unique_oris])

    if normalize:
        counts = counts / len(pref_oris) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(unique_oris, counts,
           width=np.diff(unique_oris).min() * 0.8 if len(unique_oris) > 1 else 1.0,
           color=color, edgecolor='black', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Orientation (degrees)', fontsize=12, fontweight='bold')
    ylabel = 'Number of Cells (%)' if normalize else 'Number of Cells'
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n' + title
    elif probe_name:
        title = f'{probe_name}\n' + title
    elif mouse_name:
        title = f'{mouse_name}\n' + title

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(unique_oris)
    ax.set_xticklabels([f'{int(ori)}' for ori in unique_oris])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_preferred_tf_bar(peak_df,
                          peak_dff_min=1.0,
                          output_dir=None,
                          color='darkorange',
                          probe_name=None,
                          mouse_name=None,
                          normalize=False,
                          nested=False):
    """
    Plot preferred temporal frequency as a bar histogram.

    Parameters
    ----------
    nested : bool, default=False
        If True, use nested preferred TF values
    normalize : bool, default=False
        If True, normalize counts to percentages (0-100)
    """
    vis_cells = (peak_df.peak_dff_dg > peak_dff_min)

    if nested:
        pref_tfs = peak_df.loc[vis_cells].pref_tf_nested.dropna().values
        title = f'Nested Preferred Temporal Frequency Distribution\n({len(pref_tfs)} cells)'
        save_path = output_dir / 'pref_tf_distribution_nested.png' if output_dir else None
    else:
        pref_tfs = peak_df.loc[vis_cells].pref_tf.dropna().values
        title = f'Preferred Temporal Frequency Distribution\n({len(pref_tfs)} cells)'
        save_path = output_dir / 'pref_tf_distribution.png' if output_dir else None

    if len(pref_tfs) == 0:
        print("No cells with preferred temporal frequency data")
        return

    unique_tfs = np.sort(np.unique(pref_tfs))
    counts = np.array([np.sum(pref_tfs == tf) for tf in unique_tfs])

    if normalize:
        counts = counts / len(pref_tfs) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = np.arange(len(unique_tfs))
    ax.bar(x_positions, counts, width=0.8,
           color=color, edgecolor='black', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Temporal Frequency (Hz)', fontsize=12, fontweight='bold')
    ylabel = 'Number of Cells (%)' if normalize else 'Number of Cells'
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n' + title
    elif probe_name:
        title = f'{probe_name}\n' + title
    elif mouse_name:
        title = f'{mouse_name}\n' + title

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{tf:.1f}' for tf in unique_tfs])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_preferred_sf_bar(peak_df,
                          peak_dff_min=1.0,
                          output_dir=None,
                          color='orangered',
                          probe_name=None,
                          mouse_name=None,
                          normalize=False,
                          nested=False):
    """
    Plot preferred spatial frequency as a bar histogram.

    Parameters
    ----------
    nested : bool, default=False
        If True, use nested preferred SF values
    normalize : bool, default=False
        If True, normalize counts to percentages (0-100)
    """
    vis_cells = (peak_df.peak_dff_dg > peak_dff_min)

    if nested:
        pref_sfs = peak_df.loc[vis_cells].pref_sf_nested.dropna().values
        title = f'Nested Preferred Spatial Frequency Distribution\n({len(pref_sfs)} cells)'
        save_path = output_dir / 'pref_sf_distribution_nested.png' if output_dir else None
    else:
        pref_sfs = peak_df.loc[vis_cells].pref_sf.dropna().values
        title = f'Preferred Spatial Frequency Distribution\n({len(pref_sfs)} cells)'
        save_path = output_dir / 'pref_sf_distribution.png' if output_dir else None

    if len(pref_sfs) == 0:
        print("No cells with preferred spatial frequency data")
        return

    unique_sfs = np.sort(np.unique(pref_sfs))
    counts = np.array([np.sum(pref_sfs == sf) for sf in unique_sfs])

    if normalize:
        counts = counts / len(pref_sfs) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = np.arange(len(unique_sfs))
    ax.bar(x_positions, counts, width=0.8,
           color=color, edgecolor='black', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Spatial Frequency (cpd)', fontsize=12, fontweight='bold')
    ylabel = 'Number of Cells (%)' if normalize else 'Number of Cells'
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n' + title
    elif probe_name:
        title = f'{probe_name}\n' + title
    elif mouse_name:
        title = f'{mouse_name}\n' + title

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{sf:.2f}' for sf in unique_sfs])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_preferred_orientation_by_rf_from_csv(combined_df, filtered_rfs, output_dir=None,
                                              probe_name=None, mouse_name=None, background=True, binned=False):
    """
    Plot RF centers colored by preferred orientation with average RF background.
    Uses CSV data with RF centers already computed.
    
    Parameters:
    -----------
    combined_df : DataFrame
        Results dataframe with RF center positions
    filtered_rfs : list or array
        List of receptive field arrays (2D) for each unit
    output_dir : str or Path, optional
        Path to save figure
    probe_name : str, optional
        Probe name for title
    mouse_name : str, optional
        Mouse name for title
    background : bool
        Whether to show average RF as background
    binned : bool
        Whether to bin units into 5x5 degree squares and show mode orientation
    """
    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center'])
    
    if len(df) == 0:
        print("No valid RF center data found")
        return
    
    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    preferred_oris = df['pref_ori'].values
    
    unique_oris = np.unique(preferred_oris)
    colors = ListedColormap(plt.cm.tab10.colors[:len(unique_oris)])
    ori_to_idx = {ori: i for i, ori in enumerate(unique_oris)}

    title = f'RF Centers Colored by {"Binned " if binned else ""}Preferred Orientation\n({len(df)} cells)'
    filename = 'rf_centers_by_preferred_orientation_binned' if binned else 'rf_centers_by_preferred_orientation'
    save_path = output_dir / f'{filename}.png' if output_dir else None

    x_min, x_max = -40, 40
    y_min, y_max = -40, 40
    x_padding = 0
    y_padding = 0

    fig, ax = plt.subplots(figsize=(12, 10))

    if filtered_rfs is not None and len(filtered_rfs) > 0:
        average_rf = np.mean(filtered_rfs, axis=0)
        if background:
            ax.imshow(average_rf, origin="lower", cmap='viridis', alpha=0.4,
                      extent=[x_min - x_padding, x_max + x_padding,
                               y_min - y_padding, y_max + y_padding],
                      aspect='auto')

    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    if binned:
        bin_size = 5
        x_bins = np.arange(x_min, x_max + bin_size, bin_size)
        y_bins = np.arange(y_min, y_max + bin_size, bin_size)

        bin_x_centers = []
        bin_y_centers = []
        bin_mode_ori = []

        for x_start in x_bins[:-1]:
            for y_start in y_bins[:-1]:
                in_bin = (
                    (x_positions >= x_start) & (x_positions < x_start + bin_size) &
                    (y_positions >= y_start) & (y_positions < y_start + bin_size)
                )
                if in_bin.sum() > 0:
                    bin_x_centers.append(x_start + bin_size / 2)
                    bin_y_centers.append(y_start + bin_size / 2)
                    bin_mode_ori.append(pd.Series(preferred_oris[in_bin]).mode()[0])

        for ori in unique_oris:
            mask = np.array(bin_mode_ori) == ori
            if mask.sum() > 0:
                ax.scatter(np.array(bin_x_centers)[mask], np.array(bin_y_centers)[mask],
                           color=colors(ori_to_idx[ori]), label=f'{int(ori)}°',
                           s=200, alpha=0.9, edgecolors='black', linewidths=1.0)
    else:
        for ori in unique_oris:
            ori_mask = (preferred_oris == ori)
            ax.scatter(x_positions[ori_mask], y_positions[ori_mask],
                       color=colors(ori_to_idx[ori]), label=f'{int(ori)}°',
                       s=100, alpha=0.8, edgecolors='black', linewidths=1.0)

    ax.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax.set_ylabel('RF Center Y Position (degrees)', fontsize=14)

    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n' + title
    elif probe_name:
        title = f'{probe_name}\n' + title
    elif mouse_name:
        title = f'{mouse_name}\n' + title

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(title='Preferred Orientation', bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

def plot_osi_by_rf_from_csv(combined_df, filtered_rfs, output_dir=None,
                                     probe_name=None, mouse_name=None, background=True, nested=False, binned=False):
    """
    Plot RF centers colored by osi with average RF background.
    Uses CSV data with RF centers already computed.
    
    Parameters:
    -----------
    combined_df : DataFrame
        Results dataframe with RF center positions
    filtered_rfs : list or array
        List of receptive field arrays (2D) for each unit
    output_dir : str or Path, optional
        Path to save figure
    probe_name : str, optional
        Probe name for title
    mouse_name : str, optional
        Mouse name for title
    background : bool
        Whether to show average RF as background
    nested : bool
        Whether to use nested OSI values
    binned : bool
        Whether to bin units into 5x5 degree squares and show average OSI
    """
    # Filter out rows without RF center data
    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center'])
    
    if len(df) == 0:
        print("No valid RF center data found")
        return
    
    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    
    if nested:
        osis = df['osi_dg_nested'].values
        title = f'RF Centers Colored by Nested {"Binned " if binned else ""}OSI\n({len(df)} cells)'
        suffix = 'osi_nested'
    else:
        osis = df['osi_dg'].values
        title = f'RF Centers Colored by {"Binned " if binned else ""}OSI\n({len(df)} cells)'
        suffix = 'osi'

    # Build save path based on flags
    filename = f'rf_centers_by_{suffix}'
    if binned:
        filename += '_binned'
    save_path = output_dir / f'{filename}.png' if output_dir else None

    # Set up colormap — continuous for binned, categorical otherwise
    x_min, x_max = -40, 40
    y_min, y_max = -40, 40
    x_padding = 0
    y_padding = 0

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate and plot average RF as background
    if filtered_rfs is not None and len(filtered_rfs) > 0:
        average_rf = np.mean(filtered_rfs, axis=0)
        if background:
            ax.imshow(average_rf, origin="lower", cmap='viridis', alpha=0.4, 
                        extent=[x_min - x_padding, x_max + x_padding, 
                                y_min - y_padding, y_max + y_padding],
                        aspect='auto')
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    if binned:
        # Bin units into 5x5 degree squares and compute mean OSI per bin
        bin_size = 5
        x_bins = np.arange(x_min, x_max + bin_size, bin_size)
        y_bins = np.arange(y_min, y_max + bin_size, bin_size)

        bin_x_centers = []
        bin_y_centers = []
        bin_mean_osi = []

        for x_start in x_bins[:-1]:
            for y_start in y_bins[:-1]:
                in_bin = (
                    (x_positions >= x_start) & (x_positions < x_start + bin_size) &
                    (y_positions >= y_start) & (y_positions < y_start + bin_size)
                )
                if in_bin.sum() > 0:
                    bin_x_centers.append(x_start + bin_size / 2)
                    bin_y_centers.append(y_start + bin_size / 2)
                    bin_mean_osi.append(np.mean(osis[in_bin]))

        bin_x_centers = np.array(bin_x_centers)
        bin_y_centers = np.array(bin_y_centers)
        bin_mean_osi = np.array(bin_mean_osi)

        # Continuous colormap for OSI (0 to 1)
        cmap = plt.cm.Spectral
        sc = ax.scatter(bin_x_centers, bin_y_centers,
                        c=bin_mean_osi, cmap=cmap, vmin=0, vmax=1,
                        s=200, alpha=0.9, edgecolors='black', linewidths=1.0)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Mean OSI', fontsize=12)

    else:
        # Continuous colormap for individual units
        cmap = plt.cm.Spectral
        sc = ax.scatter(x_positions, y_positions,
                        c=osis, cmap=cmap, vmin=0, vmax=1,
                        s=100, alpha=0.8, edgecolors='black', linewidths=1.0)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('OSI', fontsize=12)
    
    ax.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax.set_ylabel('RF Center Y Position (degrees)', fontsize=14)
    
    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n' + title
    elif probe_name:
        title = f'{probe_name}\n' + title
    elif mouse_name:
        title = f'{mouse_name}\n' + title
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_preferred_tf_by_rf_from_csv(combined_df, filtered_rfs, output_dir=None,
                                     probe_name=None, mouse_name=None, background=True, nested=False, binned=False):
    """
    Plot RF centers colored by preferred temporal frequency with average RF background.
    Uses CSV data with RF centers already computed.
    
    Parameters:
    -----------
    combined_df : DataFrame
        Results dataframe with RF center positions
    filtered_rfs : list or array
        List of receptive field arrays (2D) for each unit
    output_dir : str or Path, optional
        Path to save figure
    probe_name : str, optional
        Probe name for title
    mouse_name : str, optional
        Mouse name for title
    background : bool
        Whether to show average RF as background
    nested : bool
        Whether to use nested TF values
    binned : bool
        Whether to bin units into 5x5 degree squares and show mode TF
    """
    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center'])
    
    if len(df) == 0:
        print("No valid RF center data found")
        return
    
    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    
    if nested:
        preferred_tfs = df['pref_tf_nested'].values
        title = f'RF Centers Colored by Nested {"Binned " if binned else ""}Preferred Temporal Frequency\n({len(df)} cells)'
        suffix = 'tf_nested'
    else:
        preferred_tfs = df['pref_tf'].values
        title = f'RF Centers Colored by {"Binned " if binned else ""}Preferred Temporal Frequency\n({len(df)} cells)'
        suffix = 'tf'

    filename = f'rf_centers_by_preferred_{suffix}'
    if binned:
        filename += '_binned'
    save_path = output_dir / f'{filename}.png' if output_dir else None

    unique_tfs = np.unique(preferred_tfs)
    colors = ListedColormap(plt.cm.tab10.colors[:len(unique_tfs)])
    tf_to_idx = {tf: i for i, tf in enumerate(unique_tfs)}

    x_min, x_max = -40, 40
    y_min, y_max = -40, 40
    x_padding = 0
    y_padding = 0

    fig, ax = plt.subplots(figsize=(12, 10))

    if filtered_rfs is not None and len(filtered_rfs) > 0:
        average_rf = np.mean(filtered_rfs, axis=0)
        if background:
            ax.imshow(average_rf, origin="lower", cmap='viridis', alpha=0.4,
                      extent=[x_min - x_padding, x_max + x_padding,
                               y_min - y_padding, y_max + y_padding],
                      aspect='auto')

    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    if binned:
        bin_size = 5
        x_bins = np.arange(x_min, x_max + bin_size, bin_size)
        y_bins = np.arange(y_min, y_max + bin_size, bin_size)

        bin_x_centers = []
        bin_y_centers = []
        bin_mode_tf = []

        for x_start in x_bins[:-1]:
            for y_start in y_bins[:-1]:
                in_bin = (
                    (x_positions >= x_start) & (x_positions < x_start + bin_size) &
                    (y_positions >= y_start) & (y_positions < y_start + bin_size)
                )
                if in_bin.sum() > 0:
                    bin_x_centers.append(x_start + bin_size / 2)
                    bin_y_centers.append(y_start + bin_size / 2)
                    mode_tf = pd.Series(preferred_tfs[in_bin]).mode()[0]
                    bin_mode_tf.append(mode_tf)

        # Plot binned points, one scatter call per unique TF for clean legend
        for tf in unique_tfs:
            mask = np.array(bin_mode_tf) == tf
            if mask.sum() > 0:
                ax.scatter(np.array(bin_x_centers)[mask], np.array(bin_y_centers)[mask],
                           color=colors(tf_to_idx[tf]), label=f'{tf:.1f} Hz',
                           s=200, alpha=0.9, edgecolors='black', linewidths=1.0)
    else:
        for tf in unique_tfs:
            tf_mask = (preferred_tfs == tf)
            ax.scatter(x_positions[tf_mask], y_positions[tf_mask],
                       color=colors(tf_to_idx[tf]), label=f'{tf:.1f} Hz',
                       s=100, alpha=0.8, edgecolors='black', linewidths=1.0)

    ax.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax.set_ylabel('RF Center Y Position (degrees)', fontsize=14)

    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n' + title
    elif probe_name:
        title = f'{probe_name}\n' + title
    elif mouse_name:
        title = f'{mouse_name}\n' + title

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(title='Preferred TF', bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_preferred_sf_by_rf_from_csv(combined_df, filtered_rfs, output_dir=None,
                                     probe_name=None, mouse_name=None, background=True, nested=False, binned=False):
    """
    Plot RF centers colored by preferred spatial frequency with average RF background.
    Uses CSV data with RF centers already computed.
    
    Parameters:
    -----------
    combined_df : DataFrame
        Results dataframe with RF center positions
    filtered_rfs : list or array
        List of receptive field arrays (2D) for each unit
    output_dir : str or Path, optional
        Path to save figure
    probe_name : str, optional
        Probe name for title
    mouse_name : str, optional
        Mouse name for title
    background : bool
        Whether to show average RF as background
    nested : bool
        Whether to use nested SF values
    binned : bool
        Whether to bin units into 5x5 degree squares and show mode SF
    """
    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center'])
    
    if len(df) == 0:
        print("No valid RF center data found")
        return
    
    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values

    if nested:
        preferred_sfs = df['pref_sf_nested'].values
        title = f'RF Centers Colored by Nested {"Binned " if binned else ""}Preferred Spatial Frequency\n({len(df)} cells)'
        suffix = 'sf_nested'
    else:
        preferred_sfs = df['pref_sf'].values
        title = f'RF Centers Colored by {"Binned " if binned else ""}Preferred Spatial Frequency\n({len(df)} cells)'
        suffix = 'sf'

    filename = f'rf_centers_by_preferred_{suffix}'
    if binned:
        filename += '_binned'
    save_path = output_dir / f'{filename}.png' if output_dir else None

    unique_sfs = np.unique(preferred_sfs)
    colors = ListedColormap(plt.cm.tab10.colors[:len(unique_sfs)])
    sf_to_idx = {sf: i for i, sf in enumerate(unique_sfs)}

    x_min, x_max = -40, 40
    y_min, y_max = -40, 40
    x_padding = 0
    y_padding = 0

    fig, ax = plt.subplots(figsize=(12, 10))

    if filtered_rfs is not None and len(filtered_rfs) > 0:
        average_rf = np.mean(filtered_rfs, axis=0)
        if background:
            ax.imshow(average_rf, origin="lower", cmap='viridis', alpha=0.4,
                      extent=[x_min - x_padding, x_max + x_padding,
                               y_min - y_padding, y_max + y_padding],
                      aspect='auto')

    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    if binned:
        bin_size = 5
        x_bins = np.arange(x_min, x_max + bin_size, bin_size)
        y_bins = np.arange(y_min, y_max + bin_size, bin_size)

        bin_x_centers = []
        bin_y_centers = []
        bin_mode_sf = []

        for x_start in x_bins[:-1]:
            for y_start in y_bins[:-1]:
                in_bin = (
                    (x_positions >= x_start) & (x_positions < x_start + bin_size) &
                    (y_positions >= y_start) & (y_positions < y_start + bin_size)
                )
                if in_bin.sum() > 0:
                    bin_x_centers.append(x_start + bin_size / 2)
                    bin_y_centers.append(y_start + bin_size / 2)
                    mode_sf = pd.Series(preferred_sfs[in_bin]).mode()[0]
                    bin_mode_sf.append(mode_sf)

        # Plot binned points, one scatter call per unique SF for clean legend
        for sf in unique_sfs:
            mask = np.array(bin_mode_sf) == sf
            if mask.sum() > 0:
                ax.scatter(np.array(bin_x_centers)[mask], np.array(bin_y_centers)[mask],
                           color=colors(sf_to_idx[sf]), label=f'{sf:.2f} cpd',
                           s=200, alpha=0.9, edgecolors='black', linewidths=1.0)
    else:
        for sf in unique_sfs:
            sf_mask = (preferred_sfs == sf)
            ax.scatter(x_positions[sf_mask], y_positions[sf_mask],
                       color=colors(sf_to_idx[sf]), label=f'{sf:.2f} cpd',
                       s=100, alpha=0.8, edgecolors='black', linewidths=1.0)

    ax.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax.set_ylabel('RF Center Y Position (degrees)', fontsize=14)

    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n' + title
    elif probe_name:
        title = f'{probe_name}\n' + title
    elif mouse_name:
        title = f'{mouse_name}\n' + title

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(title='Preferred SF', bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_rf_position_vs_pref_ori_from_csv(combined_df, save_path_x=None, save_path_y=None,
                                          probe_name=None, mouse_name=None):
    """
    Create scatter plots: RF position (X and Y) vs preferred orientation.
    Uses CSV data with RF centers already computed.
    """
    # Filter out rows without RF center data
    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center'])
    
    if len(df) == 0:
        print("No valid RF center data found")
        return
    
    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    preferred_oris = df['pref_ori'].values
    
    unique_oris = np.unique(preferred_oris)
    n_orientations = len(unique_oris)
    
    if n_orientations <= 10:
        ori_colors = plt.cm.tab10(np.linspace(0, 1, n_orientations))
    else:
        ori_colors = plt.cm.tab20(np.linspace(0, 1, n_orientations))
    
    base_title_suffix = f'\n({len(df)} cells)'
    if mouse_name and probe_name:
        base_title = f'{mouse_name} - {probe_name}'
    elif probe_name:
        base_title = f'{probe_name}'
    elif mouse_name:
        base_title = f'{mouse_name}'
    else:
        base_title = ''
    
    # Plot 1: X position vs Preferred Orientation
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    for i, ori in enumerate(unique_oris):
        ori_mask = (preferred_oris == ori)
        ax1.scatter(x_positions[ori_mask], preferred_oris[ori_mask], 
                   color=ori_colors[i], label=f'{int(ori)}°', s=100, alpha=0.7)
    
    ax1.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax1.set_ylabel('Preferred Orientation (degrees)', fontsize=14)
    
    title1 = 'RF X Position vs Preferred Orientation'
    if base_title:
        title1 = base_title + '\n' + title1 + base_title_suffix
    else:
        title1 = title1 + base_title_suffix
    
    ax1.set_title(title1, fontsize=16, fontweight='bold', pad=20)
    ax1.legend(title='Preferred Orientation', bbox_to_anchor=(1.05, 1), 
              loc='upper left', fontsize=10, title_fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path_x:
        plt.savefig(save_path_x, dpi=300, bbox_inches='tight')
        #print(f"✓ Saved X position figure to: {save_path_x}")
    
    plt.close()
    
    # Plot 2: Y position vs Preferred Orientation
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    for i, ori in enumerate(unique_oris):
        ori_mask = (preferred_oris == ori)
        ax2.scatter(y_positions[ori_mask], preferred_oris[ori_mask], 
                   color=ori_colors[i], label=f'{int(ori)}°', s=100, alpha=0.7)
    
    ax2.set_xlabel('RF Center Y Position (degrees)', fontsize=14)
    ax2.set_ylabel('Preferred Orientation (degrees)', fontsize=14)
    
    title2 = 'RF Y Position vs Preferred Orientation'
    if base_title:
        title2 = base_title + '\n' + title2 + base_title_suffix
    else:
        title2 = title2 + base_title_suffix
    
    ax2.set_title(title2, fontsize=16, fontweight='bold', pad=20)
    ax2.legend(title='Preferred Orientation', bbox_to_anchor=(1.05, 1), 
              loc='upper left', fontsize=10, title_fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path_y:
        plt.savefig(save_path_y, dpi=300, bbox_inches='tight')
        #print(f"✓ Saved Y position figure to: {save_path_y}")
    
    plt.close()


def plot_rf_position_vs_pref_tf_from_csv(combined_df, save_path_x=None, save_path_y=None,
                                         probe_name=None, mouse_name=None):
    """
    Create scatter plots: RF position (X and Y) vs preferred temporal frequency.
    Uses CSV data with RF centers already computed.
    """
    # Filter out rows without RF center data
    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center'])
    
    if len(df) == 0:
        print("No valid RF center data found")
        return
    
    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    preferred_tfs = df['pref_tf'].values
    
    unique_tfs = np.unique(preferred_tfs)
    n_tfs = len(unique_tfs)
    
    if n_tfs <= 10:
        tf_colors = plt.cm.tab10(np.linspace(0, 1, n_tfs))
    else:
        tf_colors = plt.cm.tab20(np.linspace(0, 1, n_tfs))
    
    base_title_suffix = f'\n({len(df)} cells)'
    if mouse_name and probe_name:
        base_title = f'{mouse_name} - {probe_name}'
    elif probe_name:
        base_title = f'{probe_name}'
    elif mouse_name:
        base_title = f'{mouse_name}'
    else:
        base_title = ''
    
    # Plot 1: X position vs Preferred TF
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    for i, tf in enumerate(unique_tfs):
        tf_mask = (preferred_tfs == tf)
        ax1.scatter(x_positions[tf_mask], preferred_tfs[tf_mask], 
                   color=tf_colors[i], label=f'{tf:.1f} Hz', s=100, alpha=0.7)
    
    ax1.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax1.set_ylabel('Preferred Temporal Frequency (Hz)', fontsize=14)
    
    title1 = 'RF X Position vs Preferred Temporal Frequency'
    if base_title:
        title1 = base_title + '\n' + title1 + base_title_suffix
    else:
        title1 = title1 + base_title_suffix
    
    ax1.set_title(title1, fontsize=16, fontweight='bold', pad=20)
    ax1.legend(title='Preferred TF', bbox_to_anchor=(1.05, 1), 
              loc='upper left', fontsize=10, title_fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path_x:
        plt.savefig(save_path_x, dpi=300, bbox_inches='tight')
        #print(f"✓ Saved X position figure to: {save_path_x}")
    
    plt.close()
    
    # Plot 2: Y position vs Preferred TF
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    for i, tf in enumerate(unique_tfs):
        tf_mask = (preferred_tfs == tf)
        ax2.scatter(y_positions[tf_mask], preferred_tfs[tf_mask], 
                   color=tf_colors[i], label=f'{tf:.1f} Hz', s=100, alpha=0.7)
    
    ax2.set_xlabel('RF Center Y Position (degrees)', fontsize=14)
    ax2.set_ylabel('Preferred Temporal Frequency (Hz)', fontsize=14)
    
    title2 = 'RF Y Position vs Preferred Temporal Frequency'
    if base_title:
        title2 = base_title + '\n' + title2 + base_title_suffix
    else:
        title2 = title2 + base_title_suffix
    
    ax2.set_title(title2, fontsize=16, fontweight='bold', pad=20)
    ax2.legend(title='Preferred TF', bbox_to_anchor=(1.05, 1), 
              loc='upper left', fontsize=10, title_fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path_y:
        plt.savefig(save_path_y, dpi=300, bbox_inches='tight')
        #print(f"✓ Saved Y position figure to: {save_path_y}")
    
    plt.close()


def plot_rf_position_vs_pref_sf_from_csv(combined_df, save_path_x=None, save_path_y=None,
                                         probe_name=None, mouse_name=None):
    """
    Create scatter plots: RF position (X and Y) vs preferred spatial frequency.
    Uses CSV data with RF centers already computed.
    """
    # Filter out rows without RF center data
    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center'])
    
    if len(df) == 0:
        print("No valid RF center data found")
        return
    
    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    preferred_sfs = df['pref_sf'].values
    
    unique_sfs = np.unique(preferred_sfs)
    n_sfs = len(unique_sfs)
    
    if n_sfs <= 10:
        sf_colors = plt.cm.tab10(np.linspace(0, 1, n_sfs))
    else:
        sf_colors = plt.cm.tab20(np.linspace(0, 1, n_sfs))
    
    base_title_suffix = f'\n({len(df)} cells)'
    if mouse_name and probe_name:
        base_title = f'{mouse_name} - {probe_name}'
    elif probe_name:
        base_title = f'{probe_name}'
    elif mouse_name:
        base_title = f'{mouse_name}'
    else:
        base_title = ''
    
    # Plot 1: X position vs Preferred SF
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    for i, sf in enumerate(unique_sfs):
        sf_mask = (preferred_sfs == sf)
        ax1.scatter(x_positions[sf_mask], preferred_sfs[sf_mask], 
                   color=sf_colors[i], label=f'{sf:.2f} cpd', s=100, alpha=0.7)
    
    ax1.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax1.set_ylabel('Preferred Spatial Frequency (cpd)', fontsize=14)
    
    title1 = 'RF X Position vs Preferred Spatial Frequency'
    if base_title:
        title1 = base_title + '\n' + title1 + base_title_suffix
    else:
        title1 = title1 + base_title_suffix
    
    ax1.set_title(title1, fontsize=16, fontweight='bold', pad=20)
    ax1.legend(title='Preferred SF', bbox_to_anchor=(1.05, 1), 
              loc='upper left', fontsize=10, title_fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path_x:
        plt.savefig(save_path_x, dpi=300, bbox_inches='tight')
        #print(f"✓ Saved X position figure to: {save_path_x}")
    
    plt.close()
    
    # Plot 2: Y position vs Preferred SF
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    for i, sf in enumerate(unique_sfs):
        sf_mask = (preferred_sfs == sf)
        ax2.scatter(y_positions[sf_mask], preferred_sfs[sf_mask], 
                   color=sf_colors[i], label=f'{sf:.2f} cpd', s=100, alpha=0.7)
    
    ax2.set_xlabel('RF Center Y Position (degrees)', fontsize=14)
    ax2.set_ylabel('Preferred Spatial Frequency (cpd)', fontsize=14)
    
    title2 = 'RF Y Position vs Preferred Spatial Frequency'
    if base_title:
        title2 = base_title + '\n' + title2 + base_title_suffix
    else:
        title2 = title2 + base_title_suffix
    
    ax2.set_title(title2, fontsize=16, fontweight='bold', pad=20)
    ax2.legend(title='Preferred SF', bbox_to_anchor=(1.05, 1), 
              loc='upper left', fontsize=10, title_fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path_y:
        plt.savefig(save_path_y, dpi=300, bbox_inches='tight')
        #print(f"✓ Saved Y position figure to: {save_path_y}")
    
    plt.close()