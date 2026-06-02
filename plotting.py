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
from scipy.spatial import cKDTree
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

    # OSI distribution (non-nested, nested, and gaussian)
    for variant_kwargs in [{"nested": False}, {"nested": True}, {"gaussian": True}]:
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

    # Preferred TF and SF bar charts (non-nested, nested, gaussian, and gaussian snapped)
    bar_configs = [
        (plot_preferred_tf_bar, [
            {"nested": False},
            {"nested": True},
            {"gaussian": True},
            {"gaussian": True, "snapped": True},
        ]),
        (plot_preferred_sf_bar, [
            {"nested": False},
            {"nested": True},
            {"gaussian": True},
            {"gaussian": True, "snapped": True},
        ]),
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
                {"gaussian": True},
                {"nested": False, "binned": True},
                {"nested": True,  "binned": True},
                {"gaussian": True, "binned": True},
            ]),
            (plot_preferred_tf_by_rf_from_csv, [
                {"nested": False},
                {"nested": True},
                {"gaussian": True},
                {"gaussian": True, "snapped": True},
                {"nested": False, "binned": True},
                {"nested": True,  "binned": True},
                {"gaussian": True, "binned": True},
                {"gaussian": True, "snapped": True, "binned": True},
            ]),
            (plot_preferred_sf_by_rf_from_csv, [
                {"nested": False},
                {"nested": True},
                {"gaussian": True},
                {"gaussian": True, "snapped": True},
                {"nested": False, "binned": True},
                {"nested": True,  "binned": True},
                {"gaussian": True, "binned": True},
                {"gaussian": True, "snapped": True, "binned": True},
            ]),
        ]

        for plot_fn, variants in rf_configs:
            for variant_kwargs in variants:
                plot_fn(df, filtered_rfs, **rf_shared_kwargs, **variant_kwargs)

        # FWHM heatmap
        if 'rf_fwhm_deg' in df.columns:
            plot_rf_fwhm_heatmap(
                df,
                output_dir=output_dir,
                probe_name=probe_name,
                mouse_name=mouse_name,
            )

        # Scatter heatmap
        plot_rf_scatter_heatmap(
            df,
            output_dir=output_dir,
            probe_name=probe_name,
            mouse_name=mouse_name,
        )


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
        plot_configs = [
            (plot_preferred_orientation_by_rf_from_csv, [
                {"binned": True},
                {"binned": False},
            ]),
            (plot_osi_by_rf_from_csv, [
                {"nested": False},
                {"nested": True},
                {"gaussian": True},
                {"nested": False, "binned": True},
                {"nested": True,  "binned": True},
                {"gaussian": True, "binned": True},
            ]),
            (plot_preferred_tf_by_rf_from_csv, [
                {"nested": False},
                {"nested": True},
                {"gaussian": True},
                {"gaussian": True, "snapped": True},
                {"nested": False, "binned": True},
                {"nested": True,  "binned": True},
                {"gaussian": True, "binned": True},
                {"gaussian": True, "snapped": True, "binned": True},
            ]),
            (plot_preferred_sf_by_rf_from_csv, [
                {"nested": False},
                {"nested": True},
                {"gaussian": True},
                {"gaussian": True, "snapped": True},
                {"nested": False, "binned": True},
                {"nested": True,  "binned": True},
                {"gaussian": True, "binned": True},
                {"gaussian": True, "snapped": True, "binned": True},
            ]),
        ]

        shared_kwargs = dict(
            output_dir=output_dir,
            probe_name=probe_name,
            mouse_name=mouse_name,
            background=False,
        )

        for plot_fn, variants in plot_configs:
            for variant_kwargs in variants:
                plot_fn(df, filtered_rfs, **shared_kwargs, **variant_kwargs)
        
        plot_rf_position_vs_pref_ori_from_csv(
            df,
            save_path_x=output_dir / 'rf_x_position_vs_pref_ori.png',
            save_path_y=output_dir / 'rf_y_position_vs_pref_ori.png',
            probe_name=probe_name,
            mouse_name=mouse_name
        )
        
        plot_rf_position_vs_pref_tf_from_csv(
            df,
            save_path_x=output_dir / 'rf_x_position_vs_pref_tf.png',
            save_path_y=output_dir / 'rf_y_position_vs_pref_tf.png',
            probe_name=probe_name,
            mouse_name=mouse_name
        )
        
        plot_rf_position_vs_pref_sf_from_csv(
            df,
            save_path_x=output_dir / 'rf_x_position_vs_pref_sf.png',
            save_path_y=output_dir / 'rf_y_position_vs_pref_sf.png',
            probe_name=probe_name,
            mouse_name=mouse_name
        )

        if 'rf_fwhm_deg' in df.columns:
            plot_rf_fwhm_heatmap(
                df,
                output_dir=output_dir,
                probe_name=probe_name,
                mouse_name=mouse_name,
            )

        plot_rf_scatter_heatmap(
            df,
            output_dir=output_dir,
            probe_name=probe_name,
            mouse_name=mouse_name,
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
    df = df.dropna(subset=['rf_x_center', 'rf_y_center'])
    
    if len(df) == 0:
        print("No valid RF center data found")
        return
    
    if filtered_rfs is None or len(filtered_rfs) == 0:
        print("No RFs available to plot")
        return
    
    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    average_rf = np.mean(filtered_rfs, axis=0)
    
    x_min, x_max = -40, 40
    y_min, y_max = -40, 40
    
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    im = ax.imshow(average_rf, origin="lower", cmap='viridis', alpha=0.8, 
                   extent=[x_min - x_padding, x_max + x_padding, 
                           y_min - y_padding, y_max + y_padding],
                   aspect='auto')
    
    plt.colorbar(im, ax=ax, label='Average Response')
    
    ax.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax.set_ylabel('RF Center Y Position (degrees)', fontsize=14)
    
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
                                 nested=False,
                                 gaussian=False):
    """
    Plot orientation selectivity histogram.

    Parameters
    ----------
    nested : bool, default=False
        If True, use nested OSI values (ignored if gaussian=True).
    gaussian : bool, default=False
        If True, use gaussian-fit OSI values. Takes priority over nested.
    """
    vis_cells = (peak_df.peak_dff_dg > peak_dff_min)

    if gaussian:
        col = 'osi_dg_gaussian'
        if col not in peak_df.columns:
            print(f"Column '{col}' not found, skipping gaussian OSI plot")
            return
        osi_cells = vis_cells & (peak_df[col] > si_range[0]) & (peak_df[col] < si_range[1])
        osis = peak_df.loc[osi_cells][col].values
        title = f'Gaussian Orientation Selectivity Distribution\n({len(osis)} cells)'
        save_path = output_dir / 'osi_distribution_gaussian.png' if output_dir else None
    elif nested:
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
    vis_cells = (peak_df.peak_dff_dg > peak_dff_min)
    dsi_cells = vis_cells & (peak_df.dsi_dg > si_range[0]) & (peak_df.dsi_dg < si_range[1])
    
    peak_dsi = peak_df.loc[dsi_cells]
    dsis = peak_dsi.dsi_dg.values
    
    if len(dsis) == 0:
        print("No cells passed DSI filter")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(dsis, bins=n_hist_bins, edgecolor='black', alpha=0.7, 
            cumulative=False, density=density, color='green')
    ax.set_xlabel('Direction Selectivity Index (DSI)', fontsize=12)
    ylabel = 'Number of Cells (Normalized)' if density else 'Number of Cells'
    ax.set_ylabel(ylabel, fontsize=12)
    
    title = f'Direction Selectivity Distribution\n({len(dsis)} cells)'
    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n' + title
    elif probe_name:
        title = f'{probe_name}\n' + title
    elif mouse_name:
        title = f'{mouse_name}\n' + title
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    stats_text = f'Mean: {np.mean(dsis):.3f}\nMedian: {np.median(dsis):.3f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
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
                          nested=False,
                          gaussian=False,
                          snapped=False):
    """
    Plot preferred temporal frequency as a bar histogram.

    Parameters
    ----------
    nested : bool, default=False
        If True, use nested preferred TF values (ignored if gaussian=True).
    gaussian : bool, default=False
        If True, use gaussian-fit preferred TF values. Takes priority over nested.
    snapped : bool, default=False
        If True (and gaussian=True), use gaussian values snapped to nearest
        presented stimulus value (pref_tf_gaussian_snapped).
    """
    vis_cells = (peak_df.peak_dff_dg > peak_dff_min)

    if gaussian:
        if snapped:
            col = 'pref_tf_gaussian_snapped'
            title_prefix = 'Gaussian (Snapped) Preferred Temporal Frequency Distribution'
            filename = 'pref_tf_distribution_gaussian_snapped.png'
        else:
            col = 'pref_tf_gaussian'
            title_prefix = 'Gaussian Preferred Temporal Frequency Distribution'
            filename = 'pref_tf_distribution_gaussian.png'
        if col not in peak_df.columns:
            print(f"Column '{col}' not found, skipping plot")
            return
        pref_tfs = peak_df.loc[vis_cells][col].dropna().values
        save_path = output_dir / filename if output_dir else None
    elif nested:
        pref_tfs = peak_df.loc[vis_cells].pref_tf_nested.dropna().values
        title_prefix = 'Nested Preferred Temporal Frequency Distribution'
        save_path = output_dir / 'pref_tf_distribution_nested.png' if output_dir else None
    else:
        pref_tfs = peak_df.loc[vis_cells].pref_tf.dropna().values
        title_prefix = 'Preferred Temporal Frequency Distribution'
        save_path = output_dir / 'pref_tf_distribution.png' if output_dir else None

    if len(pref_tfs) == 0:
        print("No cells with preferred temporal frequency data")
        return

    title = f'{title_prefix}\n({len(pref_tfs)} cells)'

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
                          nested=False,
                          gaussian=False,
                          snapped=False):
    """
    Plot preferred spatial frequency as a bar histogram.

    Parameters
    ----------
    nested : bool, default=False
        If True, use nested preferred SF values (ignored if gaussian=True).
    gaussian : bool, default=False
        If True, use gaussian-fit preferred SF values. Takes priority over nested.
    snapped : bool, default=False
        If True (and gaussian=True), use gaussian values snapped to nearest
        presented stimulus value (pref_sf_gaussian_snapped).
    """
    vis_cells = (peak_df.peak_dff_dg > peak_dff_min)

    if gaussian:
        if snapped:
            col = 'pref_sf_gaussian_snapped'
            title_prefix = 'Gaussian (Snapped) Preferred Spatial Frequency Distribution'
            filename = 'pref_sf_distribution_gaussian_snapped.png'
        else:
            col = 'pref_sf_gaussian'
            title_prefix = 'Gaussian Preferred Spatial Frequency Distribution'
            filename = 'pref_sf_distribution_gaussian.png'
        if col not in peak_df.columns:
            print(f"Column '{col}' not found, skipping plot")
            return
        pref_sfs = peak_df.loc[vis_cells][col].dropna().values
        save_path = output_dir / filename if output_dir else None
    elif nested:
        pref_sfs = peak_df.loc[vis_cells].pref_sf_nested.dropna().values
        title_prefix = 'Nested Preferred Spatial Frequency Distribution'
        save_path = output_dir / 'pref_sf_distribution_nested.png' if output_dir else None
    else:
        pref_sfs = peak_df.loc[vis_cells].pref_sf.dropna().values
        title_prefix = 'Preferred Spatial Frequency Distribution'
        save_path = output_dir / 'pref_sf_distribution.png' if output_dir else None

    if len(pref_sfs) == 0:
        print("No cells with preferred spatial frequency data")
        return

    title = f'{title_prefix}\n({len(pref_sfs)} cells)'

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

# TO DO: fix plotting colors: discrete not non binned, and add colorbar for binned
def plot_preferred_orientation_by_rf_from_csv(
        combined_df,
        filtered_rfs,
        output_dir=None,
        probe_name=None,
        mouse_name=None,
        background=False,
        binned=False):

    """
    Plot RF centers colored by preferred orientation with optional
    average RF background and optional spatial binning.
    """

    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center', 'pref_ori'])

    if df.empty:
        print("No valid RF center/orientation data found.")
        return

    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    preferred_oris = df['pref_ori'].values

    # --- Unique orientations & colormap ---
    unique_oris = np.sort(np.unique(preferred_oris))
    cmap = plt.cm.get_cmap('Spectral', len(unique_oris))
    ori_to_idx = {ori: i for i, ori in enumerate(unique_oris)}

    # --- Title & save path ---
    title = f'RF Centers Colored by {"Binned " if binned else ""}Preferred Orientation\n({len(df)} cells)'
    filename = 'rf_centers_by_preferred_orientation_binned' if binned else 'rf_centers_by_preferred_orientation'
    save_path = output_dir / f'{filename}.png' if output_dir else None

    # --- RF bounds ---
    x_min, x_max = -40, 40
    y_min, y_max = -40, 40

    fig, ax = plt.subplots(figsize=(12, 10))

    # --- Background average RF ---
    if background and filtered_rfs is not None and len(filtered_rfs) > 0:
        average_rf = np.mean(filtered_rfs, axis=0)
        ax.imshow(
            average_rf,
            origin="lower",
            cmap='viridis',
            alpha=0.4,
            extent=[x_min, x_max, y_min, y_max],
            aspect='auto'
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # ==========================================================
    # Binned mode
    # ==========================================================
    if binned:
        bin_size = 5
        x_bins = np.arange(x_min, x_max + bin_size, bin_size)
        y_bins = np.arange(y_min, y_max + bin_size, bin_size)

        bin_x_centers = []
        bin_y_centers = []
        bin_mean_oris = []

        for x_start in x_bins[:-1]:
            for y_start in y_bins[:-1]:
                in_bin = (
                    (x_positions >= x_start) & (x_positions < x_start + bin_size) &
                    (y_positions >= y_start) & (y_positions < y_start + bin_size)
                )
                if not np.any(in_bin):
                    continue

                bin_x_centers.append(x_start + bin_size / 2)
                bin_y_centers.append(y_start + bin_size / 2)
                bin_mean_oris.append(np.mean(preferred_oris[in_bin]))

        ori_min = np.min(unique_oris)
        ori_max = np.max(unique_oris)

        # Single scatter call + single colorbar, outside the loop
        sc = ax.scatter(
            bin_x_centers,
            bin_y_centers,
            c=bin_mean_oris,
            cmap=plt.cm.Spectral,
            vmin=ori_min,
            vmax=ori_max,
            s=220,
            alpha=0.9,
            edgecolors='black',
            linewidths=1.0
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Mean Preferred Orientation (°)', fontsize=12)

    # ==========================================================
    # Raw scatter mode
    # ==========================================================
    else:
        for ori in unique_oris:
            ori_mask = preferred_oris == ori
            ax.scatter(
                x_positions[ori_mask],
                y_positions[ori_mask],
                color=cmap(ori_to_idx[ori]),
                label=f'{int(ori)}°',
                s=100,
                alpha=0.85,
                edgecolors='black',
                linewidths=1.0
            )

            ax.legend(
            title='Preferred Orientation',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=10,
            title_fontsize=11
        )


    # --- Labels ---
    ax.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax.set_ylabel('RF Center Y Position (degrees)', fontsize=14)

    # --- Title prefix ---
    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n{title}'
    elif probe_name:
        title = f'{probe_name}\n{title}'
    elif mouse_name:
        title = f'{mouse_name}\n{title}'

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_osi_by_rf_from_csv(combined_df, filtered_rfs, output_dir=None,
                            probe_name=None, mouse_name=None, background=False,
                            nested=False, binned=False, gaussian=False):
    """
    Plot RF centers colored by OSI with average RF background.

    Parameters
    ----------
    nested : bool, default=False
        If True, use nested OSI values (ignored if gaussian=True).
    gaussian : bool, default=False
        If True, use gaussian-fit OSI values. Takes priority over nested.
    binned : bool, default=False
        If True, bin RF centers and show mean OSI per bin.
    """
    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center'])
    
    if len(df) == 0:
        print("No valid RF center data found")
        return
    
    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    
    if gaussian:
        col = 'osi_dg_gaussian'
        if col not in df.columns:
            print(f"Column '{col}' not found, skipping gaussian OSI RF plot")
            return
        osis = df[col].values
        title = f'RF Centers Colored by Gaussian {"Binned " if binned else ""}OSI\n({len(df)} cells)'
        suffix = 'osi_gaussian'
    elif nested:
        osis = df['osi_dg_nested'].values
        title = f'RF Centers Colored by Nested {"Binned " if binned else ""}OSI\n({len(df)} cells)'
        suffix = 'osi_nested'
    else:
        osis = df['osi_dg'].values
        title = f'RF Centers Colored by {"Binned " if binned else ""}OSI\n({len(df)} cells)'
        suffix = 'osi'

    filename = f'rf_centers_by_{suffix}'
    if binned:
        filename += '_binned'
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
                    bin_mean_osi.append(np.nanmean(osis[in_bin]))

        bin_x_centers = np.array(bin_x_centers)
        bin_y_centers = np.array(bin_y_centers)
        bin_mean_osi = np.array(bin_mean_osi)

        sc = ax.scatter(bin_x_centers, bin_y_centers,
                        c=bin_mean_osi, cmap=plt.cm.Spectral, vmin=0, vmax=1,
                        s=200, alpha=0.9, edgecolors='black', linewidths=1.0)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Mean OSI', fontsize=12)
    else:
        sc = ax.scatter(x_positions, y_positions,
                        c=osis, cmap=plt.cm.Spectral, vmin=0, vmax=1,
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

# TO DO: fix plotting colors: discrete not non binned, and add colorbar for binned
def plot_preferred_tf_by_rf_from_csv(combined_df, filtered_rfs, output_dir=None,
                                     probe_name=None, mouse_name=None, background=False,
                                     nested=False, binned=False, gaussian=False, snapped=False):

    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center'])

    if len(df) == 0:
        print("No valid RF center data found")
        return

    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values

    # Select correct column
    if gaussian:
        if snapped:
            col = 'pref_tf_gaussian_snapped'
            suffix = 'tf_gaussian_snapped'
        else:
            col = 'pref_tf_gaussian'
            suffix = 'tf_gaussian'
        if col not in df.columns:
            print(f"Column '{col}' not found")
            return
        preferred_tfs = df[col].values
        label = 'Preferred TF (Hz)'
    elif nested:
        preferred_tfs = df['pref_tf_nested'].values
        suffix = 'tf_nested'
        label = 'Preferred TF (Hz)'
    else:
        preferred_tfs = df['pref_tf'].values
        suffix = 'tf'
        label = 'Preferred TF (Hz)'

    filename = f'rf_centers_by_preferred_{suffix}'
    if binned:
        filename += '_binned'
    save_path = output_dir / f'{filename}.png' if output_dir else None

    x_min, x_max = -40, 40
    y_min, y_max = -40, 40

    fig, ax = plt.subplots(figsize=(12, 10))

    # Background RF
    if filtered_rfs is not None and len(filtered_rfs) > 0 and background:
        average_rf = np.mean(filtered_rfs, axis=0)
        ax.imshow(average_rf, origin="lower", cmap='viridis', alpha=0.4,
                  extent=[x_min, x_max, y_min, y_max],
                  aspect='auto')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if binned:
        bin_size = 5
        x_bins = np.arange(x_min, x_max + bin_size, bin_size)
        y_bins = np.arange(y_min, y_max + bin_size, bin_size)

        bin_x_centers = []
        bin_y_centers = []
        bin_mean_tf = []

        for x_start in x_bins[:-1]:
            for y_start in y_bins[:-1]:
                in_bin = (
                    (x_positions >= x_start) & (x_positions < x_start + bin_size) &
                    (y_positions >= y_start) & (y_positions < y_start + bin_size)
                )
                if in_bin.sum() > 0:
                    bin_x_centers.append(x_start + bin_size / 2)
                    bin_y_centers.append(y_start + bin_size / 2)
                    bin_mean_tf.append(np.nanmean(preferred_tfs[in_bin]))

        sc = ax.scatter(bin_x_centers, bin_y_centers,
                        c=bin_mean_tf,
                        cmap=plt.cm.Spectral,
                        s=200,
                        edgecolors='black',
                        linewidths=1)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(f'Mean {label}', fontsize=12)

    else:
        sc = ax.scatter(x_positions, y_positions,
                        c=preferred_tfs,
                        cmap=plt.cm.Spectral,
                        s=100,
                        edgecolors='black',
                        linewidths=1)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(label, fontsize=12)

    ax.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax.set_ylabel('RF Center Y Position (degrees)', fontsize=14)
    ax.set_title(f'RF Centers Colored by {label}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

# TO DO: fix plotting colors: discrete not non binned, and add colorbar for binned
def plot_preferred_sf_by_rf_from_csv(combined_df, filtered_rfs, output_dir=None,
                                     probe_name=None, mouse_name=None, background=False,
                                     nested=False, binned=False, gaussian=False, snapped=False):

    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center'])

    if len(df) == 0:
        print("No valid RF center data found")
        return

    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values

    # Select correct column
    if gaussian:
        if snapped:
            col = 'pref_sf_gaussian_snapped'
            suffix = 'sf_gaussian_snapped'
        else:
            col = 'pref_sf_gaussian'
            suffix = 'sf_gaussian'
        if col not in df.columns:
            print(f"Column '{col}' not found")
            return
        preferred_sfs = df[col].values
        label = 'Preferred SF (cpd)'
    elif nested:
        preferred_sfs = df['pref_sf_nested'].values
        suffix = 'sf_nested'
        label = 'Preferred SF (cpd)'
    else:
        preferred_sfs = df['pref_sf'].values
        suffix = 'sf'
        label = 'Preferred SF (cpd)'
        
    filename = f'rf_centers_by_preferred_{suffix}'
    if binned:
        filename += '_binned'
    save_path = output_dir / f'{filename}.png' if output_dir else None

    x_min, x_max = -40, 40
    y_min, y_max = -40, 40

    fig, ax = plt.subplots(figsize=(12, 10))

    # Background RF
    if filtered_rfs is not None and len(filtered_rfs) > 0 and background:
        average_rf = np.mean(filtered_rfs, axis=0)
        ax.imshow(average_rf, origin="lower", cmap='viridis', alpha=0.4,
                  extent=[x_min, x_max, y_min, y_max],
                  aspect='auto')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if binned:
        bin_size = 5
        x_bins = np.arange(x_min, x_max + bin_size, bin_size)
        y_bins = np.arange(y_min, y_max + bin_size, bin_size)

        bin_x_centers = []
        bin_y_centers = []
        bin_mean_sf = []

        for x_start in x_bins[:-1]:
            for y_start in y_bins[:-1]:
                in_bin = (
                    (x_positions >= x_start) & (x_positions < x_start + bin_size) &
                    (y_positions >= y_start) & (y_positions < y_start + bin_size)
                )
                if in_bin.sum() > 0:
                    bin_x_centers.append(x_start + bin_size / 2)
                    bin_y_centers.append(y_start + bin_size / 2)
                    bin_mean_sf.append(np.nanmean(preferred_sfs[in_bin]))

        sc = ax.scatter(bin_x_centers, bin_y_centers,
                        c=bin_mean_sf,
                        cmap=plt.cm.Spectral,
                        s=200,
                        edgecolors='black',
                        linewidths=1)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(f'Mean {label}', fontsize=12)

    else:
        sc = ax.scatter(x_positions, y_positions,
                        c=preferred_sfs,
                        cmap=plt.cm.Spectral,
                        s=100,
                        edgecolors='black',
                        linewidths=1)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(label, fontsize=12)

    ax.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax.set_ylabel('RF Center Y Position (degrees)', fontsize=14)
    ax.set_title(f'RF Centers Colored by {label}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_rf_fwhm_heatmap(combined_df, output_dir=None, probe_name=None, mouse_name=None, bin_size=5):
    """
    Plot a heatmap of mean RF FWHM (degrees) binned into spatial bins.

    Parameters:
    -----------
    combined_df : DataFrame
        Results dataframe with rf_x_center, rf_y_center, rf_fwhm_deg columns
    output_dir : Path, optional
        Output directory
    probe_name : str, optional
        Probe name for title
    mouse_name : str, optional
        Mouse name for title
    bin_size : int
        Size of spatial bins in degrees (default 5)
    """
    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center', 'rf_fwhm_deg'])

    if len(df) == 0:
        print("No valid RF FWHM data found")
        return

    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    fwhm_values = df['rf_fwhm_deg'].values

    x_min, x_max = -40, 40
    y_min, y_max = -40, 40

    x_bins = np.arange(x_min, x_max + bin_size, bin_size)
    y_bins = np.arange(y_min, y_max + bin_size, bin_size)

    heatmap = np.full((len(y_bins) - 1, len(x_bins) - 1), np.nan)
    counts = np.zeros_like(heatmap)

    for xi, x_start in enumerate(x_bins[:-1]):
        for yi, y_start in enumerate(y_bins[:-1]):
            in_bin = (
                (x_positions >= x_start) & (x_positions < x_start + bin_size) &
                (y_positions >= y_start) & (y_positions < y_start + bin_size)
            )
            if in_bin.sum() > 0:
                heatmap[yi, xi] = np.mean(fwhm_values[in_bin])
                counts[yi, xi] = in_bin.sum()

    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(
        heatmap,
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap='Spectral',
        aspect='equal',
        interpolation='nearest'
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean RF FWHM (degrees)', fontsize=12)

    x_centers = x_bins[:-1] + bin_size / 2
    y_centers = y_bins[:-1] + bin_size / 2
    for xi, xc in enumerate(x_centers):
        for yi, yc in enumerate(y_centers):
            n = int(counts[yi, xi])
            if n > 0:
                ax.text(xc, yc, str(n), ha='center', va='center',
                        fontsize=10, color='black', alpha=0.8)

    ax.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax.set_ylabel('RF Center Y Position (degrees)', fontsize=14)
    ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)

    title = f'RF Size (FWHM) by Visual Field Position\n({len(df)} cells, {bin_size}° bins)'
    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n' + title
    elif probe_name:
        title = f'{probe_name}\n' + title
    elif mouse_name:
        title = f'{mouse_name}\n' + title

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()

    save_path = output_dir / 'rf_fwhm_heatmap.png' if output_dir else None
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_rf_scatter_heatmap(combined_df, output_dir=None, probe_name=None, mouse_name=None,
                            bin_size=5, n_neighbors=5):
    """
    Estimate RF scatter as the mean distance to the N nearest neighbors in visual space,
    then plot as a heatmap binned into spatial bins.

    Note: this computes nearest neighbors in visual space (not cortical space), so it reflects
    how densely RF centers are clustered at each visual field location rather than the
    paper's cortical-space scatter measure.

    Parameters:
    -----------
    combined_df : DataFrame
        Results dataframe with rf_x_center, rf_y_center columns
    output_dir : Path, optional
        Output directory
    probe_name : str, optional
        Probe name for title
    mouse_name : str, optional
        Mouse name for title
    bin_size : int
        Size of spatial bins in degrees (default 5)
    n_neighbors : int
        Number of nearest neighbors to average distance over (default 5)
    """
    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center']).copy()

    if len(df) < n_neighbors + 1:
        print(f"Not enough units ({len(df)}) to compute {n_neighbors} nearest neighbors")
        return

    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values

    coords = np.column_stack([x_positions, y_positions])
    tree = cKDTree(coords)

    # k+1 because first result is always the point itself (distance=0)
    distances, _ = tree.query(coords, k=n_neighbors + 1)
    mean_nn_distances = distances[:, 1:].mean(axis=1)

    x_min, x_max = -40, 40
    y_min, y_max = -40, 40

    x_bins = np.arange(x_min, x_max + bin_size, bin_size)
    y_bins = np.arange(y_min, y_max + bin_size, bin_size)

    heatmap = np.full((len(y_bins) - 1, len(x_bins) - 1), np.nan)
    counts = np.zeros_like(heatmap)

    for xi, x_start in enumerate(x_bins[:-1]):
        for yi, y_start in enumerate(y_bins[:-1]):
            in_bin = (
                (x_positions >= x_start) & (x_positions < x_start + bin_size) &
                (y_positions >= y_start) & (y_positions < y_start + bin_size)
            )
            if in_bin.sum() > 0:
                heatmap[yi, xi] = mean_nn_distances[in_bin].mean()
                counts[yi, xi] = in_bin.sum()

    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(
        heatmap,
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap='Spectral',
        aspect='equal',
        interpolation='nearest'
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'Mean distance to {n_neighbors} nearest neighbors (degrees)', fontsize=12)

    x_centers = x_bins[:-1] + bin_size / 2
    y_centers = y_bins[:-1] + bin_size / 2
    for xi, xc in enumerate(x_centers):
        for yi, yc in enumerate(y_centers):
            n = int(counts[yi, xi])
            if n > 0:
                ax.text(xc, yc, str(n), ha='center', va='center',
                        fontsize=10, color='black', alpha=0.8)

    ax.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax.set_ylabel('RF Center Y Position (degrees)', fontsize=14)
    ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)

    title = f'RF Scatter ({n_neighbors}-NN mean distance) by Visual Field Position\n({len(df)} cells, {bin_size}° bins)'
    if mouse_name and probe_name:
        title = f'{mouse_name} - {probe_name}\n' + title
    elif probe_name:
        title = f'{probe_name}\n' + title
    elif mouse_name:
        title = f'{mouse_name}\n' + title

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()

    save_path = output_dir / 'rf_scatter_heatmap.png' if output_dir else None
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_rf_position_vs_pref_ori_from_csv(combined_df, save_path_x=None, save_path_y=None,
                                          probe_name=None, mouse_name=None):
    """
    Create scatter plots: RF position (X and Y) vs preferred orientation.
    """
    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center'])
    
    if len(df) == 0:
        print("No valid RF center data found")
        return
    
    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    preferred_oris = df['pref_ori'].values
    
    unique_oris = np.unique(preferred_oris)
    n_orientations = len(unique_oris)
    cmap = plt.cm.get_cmap('Spectral', n_orientations)
    ori_colors = [cmap(i) for i in range(n_orientations)]
    
    base_title_suffix = f'\n({len(df)} cells)'
    if mouse_name and probe_name:
        base_title = f'{mouse_name} - {probe_name}'
    elif probe_name:
        base_title = f'{probe_name}'
    elif mouse_name:
        base_title = f'{mouse_name}'
    else:
        base_title = ''
    
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    for i, ori in enumerate(unique_oris):
        ori_mask = (preferred_oris == ori)
        ax1.scatter(x_positions[ori_mask], preferred_oris[ori_mask],
                    color=ori_colors[i], label=f'{int(ori)}°', s=100, alpha=0.7)
    ax1.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax1.set_ylabel('Preferred Orientation (degrees)', fontsize=14)
    title1 = 'RF X Position vs Preferred Orientation'
    title1 = (base_title + '\n' + title1 + base_title_suffix) if base_title else (title1 + base_title_suffix)
    ax1.set_title(title1, fontsize=16, fontweight='bold', pad=20)
    ax1.legend(title='Preferred Orientation', bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=10, title_fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path_x:
        plt.savefig(save_path_x, dpi=300, bbox_inches='tight')
    plt.close()
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for i, ori in enumerate(unique_oris):
        ori_mask = (preferred_oris == ori)
        ax2.scatter(y_positions[ori_mask], preferred_oris[ori_mask],
                    color=ori_colors[i], label=f'{int(ori)}°', s=100, alpha=0.7)
    ax2.set_xlabel('RF Center Y Position (degrees)', fontsize=14)
    ax2.set_ylabel('Preferred Orientation (degrees)', fontsize=14)
    title2 = 'RF Y Position vs Preferred Orientation'
    title2 = (base_title + '\n' + title2 + base_title_suffix) if base_title else (title2 + base_title_suffix)
    ax2.set_title(title2, fontsize=16, fontweight='bold', pad=20)
    ax2.legend(title='Preferred Orientation', bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=10, title_fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path_y:
        plt.savefig(save_path_y, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rf_position_vs_pref_tf_from_csv(combined_df, save_path_x=None, save_path_y=None,
                                         probe_name=None, mouse_name=None):
    """
    Create scatter plots: RF position (X and Y) vs preferred temporal frequency.
    """
    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center'])
    
    if len(df) == 0:
        print("No valid RF center data found")
        return
    
    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    preferred_tfs = df['pref_tf'].values
    
    unique_tfs = np.unique(preferred_tfs)
    n_tfs = len(unique_tfs)
    cmap = plt.cm.get_cmap('Spectral', n_tfs)
    tf_colors = [cmap(i) for i in range(n_tfs)]
    
    base_title_suffix = f'\n({len(df)} cells)'
    if mouse_name and probe_name:
        base_title = f'{mouse_name} - {probe_name}'
    elif probe_name:
        base_title = f'{probe_name}'
    elif mouse_name:
        base_title = f'{mouse_name}'
    else:
        base_title = ''
    
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    for i, tf in enumerate(unique_tfs):
        tf_mask = (preferred_tfs == tf)
        ax1.scatter(x_positions[tf_mask], preferred_tfs[tf_mask],
                    color=tf_colors[i], label=f'{tf:.1f} Hz', s=100, alpha=0.7)
    ax1.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax1.set_ylabel('Preferred Temporal Frequency (Hz)', fontsize=14)
    title1 = 'RF X Position vs Preferred Temporal Frequency'
    title1 = (base_title + '\n' + title1 + base_title_suffix) if base_title else (title1 + base_title_suffix)
    ax1.set_title(title1, fontsize=16, fontweight='bold', pad=20)
    ax1.legend(title='Preferred TF', bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=10, title_fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path_x:
        plt.savefig(save_path_x, dpi=300, bbox_inches='tight')
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for i, tf in enumerate(unique_tfs):
        tf_mask = (preferred_tfs == tf)
        ax2.scatter(y_positions[tf_mask], preferred_tfs[tf_mask],
                    color=tf_colors[i], label=f'{tf:.1f} Hz', s=100, alpha=0.7)
    ax2.set_xlabel('RF Center Y Position (degrees)', fontsize=14)
    ax2.set_ylabel('Preferred Temporal Frequency (Hz)', fontsize=14)
    title2 = 'RF Y Position vs Preferred Temporal Frequency'
    title2 = (base_title + '\n' + title2 + base_title_suffix) if base_title else (title2 + base_title_suffix)
    ax2.set_title(title2, fontsize=16, fontweight='bold', pad=20)
    ax2.legend(title='Preferred TF', bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=10, title_fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path_y:
        plt.savefig(save_path_y, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rf_position_vs_pref_sf_from_csv(combined_df, save_path_x=None, save_path_y=None,
                                         probe_name=None, mouse_name=None):
    """
    Create scatter plots: RF position (X and Y) vs preferred spatial frequency.
    """
    df = combined_df.dropna(subset=['rf_x_center', 'rf_y_center'])
    
    if len(df) == 0:
        print("No valid RF center data found")
        return
    
    x_positions = df['rf_x_center'].values
    y_positions = df['rf_y_center'].values
    preferred_sfs = df['pref_sf'].values
    
    unique_sfs = np.unique(preferred_sfs)
    n_sfs = len(unique_sfs)
    cmap = plt.cm.get_cmap('Spectral', n_sfs)
    sf_colors = [cmap(i) for i in range(n_sfs)]
    
    base_title_suffix = f'\n({len(df)} cells)'
    if mouse_name and probe_name:
        base_title = f'{mouse_name} - {probe_name}'
    elif probe_name:
        base_title = f'{probe_name}'
    elif mouse_name:
        base_title = f'{mouse_name}'
    else:
        base_title = ''
    
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    for i, sf in enumerate(unique_sfs):
        sf_mask = (preferred_sfs == sf)
        ax1.scatter(x_positions[sf_mask], preferred_sfs[sf_mask],
                    color=sf_colors[i], label=f'{sf:.2f} cpd', s=100, alpha=0.7)
    ax1.set_xlabel('RF Center X Position (degrees)', fontsize=14)
    ax1.set_ylabel('Preferred Spatial Frequency (cpd)', fontsize=14)
    title1 = 'RF X Position vs Preferred Spatial Frequency'
    title1 = (base_title + '\n' + title1 + base_title_suffix) if base_title else (title1 + base_title_suffix)
    ax1.set_title(title1, fontsize=16, fontweight='bold', pad=20)
    ax1.legend(title='Preferred SF', bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=10, title_fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path_x:
        plt.savefig(save_path_x, dpi=300, bbox_inches='tight')
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for i, sf in enumerate(unique_sfs):
        sf_mask = (preferred_sfs == sf)
        ax2.scatter(y_positions[sf_mask], preferred_sfs[sf_mask],
                    color=sf_colors[i], label=f'{sf:.2f} cpd', s=100, alpha=0.7)
    ax2.set_xlabel('RF Center Y Position (degrees)', fontsize=14)
    ax2.set_ylabel('Preferred Spatial Frequency (cpd)', fontsize=14)
    title2 = 'RF Y Position vs Preferred Spatial Frequency'
    title2 = (base_title + '\n' + title2 + base_title_suffix) if base_title else (title2 + base_title_suffix)
    ax2.set_title(title2, fontsize=16, fontweight='bold', pad=20)
    ax2.legend(title='Preferred SF', bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=10, title_fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path_y:
        plt.savefig(save_path_y, dpi=300, bbox_inches='tight')
    plt.close()