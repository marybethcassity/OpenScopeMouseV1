"""
Utility functions for file management and directory organization.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import pickle


def create_output_directory(base_dir, probe, filtered=False, r2_threshold=None, curvefit=False):
    """
    Create an output directory with flags in the name.
    
    Parameters:
    -----------
    base_dir : str or Path
        Base directory for outputs
    probe : str
        Probe name
    filtered : bool
        Whether Gaussian filtering was applied
    r2_threshold : float or None
        R² threshold if filtering was applied
    curvefit : bool
        Whether curve fitting was used for preferred metrics
    
    Returns:
    --------
    Path : Created output directory path
    """
    # Build directory name with flags
    dir_parts = [probe]
    
    if filtered and r2_threshold is not None:
        dir_parts.append(f"filtered_r2-{r2_threshold:.2f}")
    elif filtered:
        dir_parts.append("filtered")
    
    # Add curvefit vs argmax
    if curvefit:
        dir_parts.append("curvefit")
    else:
        dir_parts.append("argmax")
    
    dir_name = "_".join(dir_parts)
    
    # Create full path
    output_dir = Path(base_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def save_results(df, output_dir, rf_data=None, args=None):
    """
    Save analysis results to files.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    output_dir : Path
        Output directory
    rf_data : list or None
        Raw RF data if save_raw was requested
    args : Namespace
        Command line arguments
    """
    # Save CSV
    csv_path = output_dir / 'results.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")
    
    # Save summary statistics
    summary_path = output_dir / 'summary_stats.json'
    summary = {
        'total_units': len(df),
        'num_mice': int(df['mouse_name'].nunique()),
        'mice': sorted(df['mouse_name'].unique().tolist()),
        'metrics': {}
    }
    
    for col in ['pref_ori', 'pref_tf', 'pref_sf', 'osi', 'dsi']:
        if col in df.columns:
            summary['metrics'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
    
    if 'r_squared' in df.columns:
        summary['r_squared'] = {
            'mean': float(df['r_squared'].mean()),
            'median': float(df['r_squared'].median()),
            'min': float(df['r_squared'].min()),
            'max': float(df['r_squared'].max())
        }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary statistics saved to: {summary_path}")
    
    # Save raw RF data if provided
    if rf_data is not None:
        rf_path = output_dir / 'raw_rf_data.pkl'
        with open(rf_path, 'wb') as f:
            pickle.dump(rf_data, f)
        print(f"✓ Raw RF data saved to: {rf_path}")
    
    # Save analysis parameters
    if args is not None:
        params_path = output_dir / 'analysis_parameters.json'
        params = {
            'probe': args.probe,
            'filtered': args.filtered,
            'all_units': args.all_units,
            'r2_threshold': args.r2_threshold if args.filtered else None,
            'data_dir': str(args.data_dir),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"✓ Analysis parameters saved to: {params_path}")
    
    # Create summary report
    from plotting import create_summary_report
    if args is not None:
        create_summary_report(df, output_dir, args)
        print(f"✓ Summary report saved to: {output_dir / 'analysis_summary.txt'}")


def validate_data_directory(data_dir):
    """
    Validate that data directory exists and contains mouse subdirectories.
    
    Parameters:
    -----------
    data_dir : Path
        Data directory to validate
    
    Returns:
    --------
    list : List of valid mouse directory paths
    """
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    mouse_dirs = [d for d in data_dir.iterdir() 
                  if d.is_dir() and d.name.startswith('sub-')]
    
    if not mouse_dirs:
        raise ValueError(f"No mouse directories found in {data_dir}")
    
    return mouse_dirs


def format_probe_name(probe):
    """
    Format probe name consistently.
    
    Parameters:
    -----------
    probe : str
        Probe name
    
    Returns:
    --------
    str : Formatted probe name
    """
    # Remove whitespace and ensure consistent capitalization
    probe = probe.strip()
    if not probe.startswith('Probe'):
        probe = 'Probe' + probe
    return probe


def load_previous_results(results_path):
    """
    Load previously saved results.
    
    Parameters:
    -----------
    results_path : str or Path
        Path to results CSV file
    
    Returns:
    --------
    DataFrame : Loaded results
    """
    import pandas as pd
    return pd.read_csv(results_path)


def merge_results(result_files):
    """
    Merge multiple result files.
    
    Parameters:
    -----------
    result_files : list
        List of paths to result CSV files
    
    Returns:
    --------
    DataFrame : Merged results
    """
    import pandas as pd
    dfs = [pd.read_csv(f) for f in result_files]
    return pd.concat(dfs, ignore_index=True)