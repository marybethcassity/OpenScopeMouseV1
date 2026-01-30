"""
Utility functions for file I/O and directory management.
"""

from pathlib import Path
from datetime import datetime
import pandas as pd


def create_results_directory(args):
    """
    Create a results directory with arguments encoded in the folder name.
    
    Structure: results_[probe]_[filter]_[r2]_[curvefit]_[timestamp]/
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    
    Returns:
    --------
    Path : Path to the created results directory
    """
    # Start with base directory
    base_dir = Path(args.output_dir)
    
    # Build directory name with arguments
    dir_parts = []
    
    # Add probe if specified
    if args.probe:
        dir_parts.append(f"probe-{args.probe}")
    else:
        dir_parts.append("all-probes")
    
    # Add filtering mode
    if args.filtered:
        dir_parts.append(f"filtered-r2-{args.r2_threshold:.2f}")
    else:
        dir_parts.append("all-units")
    # elif args.all_units:
    #     dir_parts.append("all-units")
    # else:
    #     dir_parts.append("snr-filtered")
    
    # Add curvefit flag if used
    if args.curvefit:
        dir_parts.append("curvefit")
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_parts.append(timestamp)
    
    # Combine parts
    dir_name = "results_" + "_".join(dir_parts)
    
    # Create full path
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def create_output_directory(base_dir, probe, filtered=False, r2_threshold=None, curvefit=False):
    """
    Create output directory for a specific probe.
    
    DEPRECATED: Use create_results_directory instead.
    This function is kept for backward compatibility.
    
    Parameters:
    -----------
    base_dir : str or Path
        Base directory for output
    probe : str
        Probe name
    filtered : bool
        Whether Gaussian filtering is applied
    r2_threshold : float, optional
        R² threshold if filtered
    curvefit : bool
        Whether curve fitting is used
    
    Returns:
    --------
    Path : Output directory path
    """
    base_dir = Path(base_dir)
    
    # Build directory name
    dir_name = probe
    
    if filtered and r2_threshold is not None:
        dir_name += f"_filtered_r2-{r2_threshold:.2f}"
    elif filtered:
        dir_name += "_filtered"
    
    if curvefit:
        dir_name += "_curvefit"
    
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def save_results(df, output_dir, rf_data=None, args=None):
    """
    Save results to CSV and optionally save raw RF data.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    output_dir : Path
        Output directory
    rf_data : list of dict, optional
        Raw receptive field data
    args : argparse.Namespace, optional
        Command line arguments
    """
    output_dir = Path(output_dir)
    
    # Save main results CSV
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved results to: {csv_path}")
    
    # Save raw RF data if provided
    if rf_data is not None and len(rf_data) > 0:
        import pickle
        rf_path = output_dir / "raw_rfs.pkl"
        with open(rf_path, 'wb') as f:
            pickle.dump(rf_data, f)
        print(f"  Saved raw RF data to: {rf_path}")
    
    # Save analysis parameters if provided
    if args is not None:
        params_path = output_dir / "analysis_params.txt"
        with open(params_path, 'w') as f:
            f.write("Analysis Parameters\n")
            f.write("=" * 50 + "\n\n")
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
        print(f"  Saved parameters to: {params_path}")


def load_results(csv_path):
    """
    Load results from CSV file.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to CSV file
    
    Returns:
    --------
    DataFrame : Results dataframe
    """
    return pd.read_csv(csv_path)


def combine_results(result_dirs):
    """
    Combine results from multiple directories.
    
    Parameters:
    -----------
    result_dirs : list of Path
        List of result directories
    
    Returns:
    --------
    DataFrame : Combined results
    """
    dfs = []
    for result_dir in result_dirs:
        csv_path = result_dir / "results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return None


def get_summary_stats(df):
    """
    Calculate summary statistics for results.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    
    Returns:
    --------
    dict : Dictionary of summary statistics
    """
    stats = {
        'n_units': len(df),
        'n_mice': df['mouse_name'].nunique() if 'mouse_name' in df.columns else None,
        'n_probes': df['probe'].nunique() if 'probe' in df.columns else None,
    }
    
    # Add metric statistics if available
    metrics = ['pref_ori', 'pref_tf', 'pref_sf', 'osi_dg', 'dsi_dg']
    for metric in metrics:
        if metric in df.columns:
            stats[f'{metric}_mean'] = df[metric].mean()
            stats[f'{metric}_median'] = df[metric].median()
            stats[f'{metric}_std'] = df[metric].std()
    
    return stats


def create_analysis_report(df, output_path, args=None):
    """
    Create a text report summarizing the analysis.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    output_path : Path
        Path to save the report
    args : argparse.Namespace, optional
        Command line arguments
    """
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PREFERRED METRICS ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Analysis parameters
        if args is not None:
            f.write("-" * 80 + "\n")
            f.write("ANALYSIS PARAMETERS\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Data directory: {args.data_dir}\n")
            f.write(f"Probe: {args.probe if args.probe else 'All probes'}\n")
            f.write(f"Filtering: ")
            if args.filtered:
                f.write(f"Gaussian fit (R² >= {args.r2_threshold})\n")
            elif args.all_units:
                f.write("All units (no filtering)\n")
            else:
                f.write("SNR > 1 only\n")
            f.write(f"Curve fitting: {'Yes' if args.curvefit else 'No'}\n")
            f.write("\n")
        
        # Summary statistics
        f.write("-" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"Total units analyzed: {len(df)}\n")
        if 'mouse_name' in df.columns:
            f.write(f"Number of mice: {df['mouse_name'].nunique()}\n")
            f.write(f"Mice: {', '.join(sorted(df['mouse_name'].unique()))}\n")
        if 'probe' in df.columns:
            f.write(f"Number of probes: {df['probe'].nunique()}\n")
            f.write(f"Probes: {', '.join(sorted(df['probe'].unique()))}\n")
        f.write("\n")
        
        # Metric statistics
        f.write("-" * 80 + "\n")
        f.write("METRIC STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        
        metrics = {
            'pref_ori': 'Preferred Orientation (degrees)',
            'pref_tf': 'Preferred Temporal Frequency (Hz)',
            'pref_sf': 'Preferred Spatial Frequency (cpd)',
            'osi_dg': 'Orientation Selectivity Index',
            'dsi_dg': 'Direction Selectivity Index',
            'peak_dff_dg': 'Peak Response'
        }
        
        for metric_col, metric_name in metrics.items():
            if metric_col in df.columns:
                f.write(f"{metric_name}:\n")
                f.write(f"  Mean: {df[metric_col].mean():.3f}\n")
                f.write(f"  Median: {df[metric_col].median():.3f}\n")
                f.write(f"  Std: {df[metric_col].std():.3f}\n")
                f.write(f"  Min: {df[metric_col].min():.3f}\n")
                f.write(f"  Max: {df[metric_col].max():.3f}\n")
                f.write("\n")
        
        # Per-mouse statistics if applicable
        if 'mouse_name' in df.columns and df['mouse_name'].nunique() > 1:
            f.write("-" * 80 + "\n")
            f.write("PER-MOUSE STATISTICS\n")
            f.write("-" * 80 + "\n\n")
            
            for mouse in sorted(df['mouse_name'].unique()):
                mouse_df = df[df['mouse_name'] == mouse]
                f.write(f"{mouse} (n={len(mouse_df)} units):\n")
                for metric_col in ['osi_dg', 'dsi_dg']:
                    if metric_col in df.columns:
                        f.write(f"  {metric_col}: mean={mouse_df[metric_col].mean():.3f}, ")
                        f.write(f"median={mouse_df[metric_col].median():.3f}\n")
                f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"  Analysis report saved to: {output_path}")