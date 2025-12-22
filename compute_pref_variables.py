#!/usr/bin/env python3
"""
Main script for analyzing preferred metrics from NWB files.

Usage:
    python compute_pref_variables.py --data_dir /path/to/data --probe ProbeA [options]
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Import custom modules
from data_processing import (
    load_nwb_data,
    process_units,
    calculate_all_metrics,
)

from gaussian_filtering import filter_rfs_by_gaussian_fit

from plotting import (
    plot_summary_figures,
    plot_rf_examples,
    plot_metric_distributions
)
from utils import create_output_directory, save_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze preferred metrics from NWB files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with filtering
  python analyze_metrics.py --data_dir /path/to/data --probe ProbeA --filtered
  
  # With custom R² threshold
  python analyze_metrics.py --data_dir /path/to/data --probe ProbeA --filtered --r2_threshold 0.7
  
  # Without filtering
  python analyze_metrics.py --data_dir /path/to/data --probe ProbeA --all_units
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to directory containing NWB files (organized by mouse)'
    )
    
    parser.add_argument(
        '--probe',
        type=str,
        default=None,
        help='Probe name to analyze (e.g., ProbeA, ProbeB). If not specified, analyzes all probes.'
    )
    
    # Filtering options
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        '--filtered',
        action='store_true',
        help='Apply Gaussian fitting filter to receptive fields'
    )

    filter_group.add_argument(
        '--all_units',
        action='store_true',
        help='Include all units without filtering'
    )
    
    parser.add_argument(
        '--r2_threshold',
        type=float,
        default=0.5,
        help='R² threshold for Gaussian fitting filter (default: 0.5)'
    )

    parser.add_argument(
        '--curvefit',
        action='store_true',
        help='Use curve fitting instead of argmax to determine preferred orientation, SF, and TF (default: False, uses argmax)'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results',
        help='Base output directory (default: ../results)'
    )
    
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    # Processing options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    parser.add_argument(
        '--save_raw',
        action='store_true',
        help='Save raw receptive field data'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Validate inputs
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    # Get list of mouse directories
    mouse_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    
    if not mouse_dirs:
        print("Error: No mouse directories found (looking for folders starting with 'sub-')")
        sys.exit(1)
    
    print(f"Found {len(mouse_dirs)} mouse directories")
    print()
    
    # Determine which probes to analyze
    if args.probe:
        probes_to_analyze = [args.probe]
        print(f"Analyzing single probe: {args.probe}")
    else:
        # Auto-detect all probes from the first mouse
        print("Auto-detecting probes...")
        nwb_path = list(mouse_dirs[0].glob("*.nwb"))[0]
        nwb_data = load_nwb_data(nwb_path)
        units = nwb_data['units']
        probes_to_analyze = sorted(units['device_name'].unique())
        print(f"Found {len(probes_to_analyze)} probes: {', '.join(probes_to_analyze)}")
    
    print()
    
    # Process each probe
    all_probes_results = []  # Store results across ALL probes for master CSV
    
    for probe_idx, probe in enumerate(probes_to_analyze, 1):
        print("=" * 80)
        print(f"Processing Probe {probe_idx}/{len(probes_to_analyze)}: {probe}")
        print("=" * 80)
        
        # Create output directory with flags in name
        output_dir = create_output_directory(
            base_dir=args.output_dir,
            probe=probe,
            filtered=args.filtered,
            r2_threshold=args.r2_threshold if args.filtered else None,
            curvefit=args.curvefit
        )
        
        print(f"Data directory: {data_dir}")
        print(f"Probe: {probe}")
        print(f"Filtering: {'Gaussian fit (R² >= ' + str(args.r2_threshold) + ')' if args.filtered else 'SNR only' if not args.all_units else 'All units'}")
        print(f"Output directory: {output_dir}")
        print()
        
        # Initialize storage for all results for this probe
        all_results = []
        all_rf_data = [] if args.save_raw else None
        
        # Process each mouse for this probe
        for mouse_idx, mouse_dir in enumerate(mouse_dirs, 1):
            mouse_name = mouse_dir.name
            print(f"  Processing mouse {mouse_idx}/{len(mouse_dirs)}: {mouse_name}")
            
            # Find NWB file
            nwb_files = list(mouse_dir.glob("*.nwb"))
            if not nwb_files:
                print(f"    Warning: No NWB file found in {mouse_dir}")
                continue
            
            nwb_path = nwb_files[0]
            
            try:
                # Load NWB data
                if args.verbose:
                    print(f"    Loading NWB file: {nwb_path.name}")
                
                nwb_data = load_nwb_data(nwb_path)
                
                # Process units and calculate receptive fields
                if args.verbose:
                    print(f"    Processing units for probe: {probe}")
                
                units_data = process_units(
                    nwb_data=nwb_data,
                    probe=probe,
                    all_units=args.all_units,
                    verbose=args.verbose
                )
                
                if units_data['unit_rfs'] is None or len(units_data['unit_rfs']) == 0:
                    print(f"    No units found for {probe}")
                    continue
                
                # Apply Gaussian filtering if requested
                if args.filtered:
                    if args.verbose:
                        print(f"    Applying Gaussian fit filter (R² >= {args.r2_threshold})")
                    
                    filtered_rfs, filtered_indices, r2_values, fitted_rfs = filter_rfs_by_gaussian_fit(
                        units_data['unit_rfs'],
                        r_squared_threshold=args.r2_threshold,
                        verbose=args.verbose
                    )
                    
                    # Update units_data with filtered results
                    units_data['unit_rfs'] = filtered_rfs
                    units_data['unit_indices'] = [units_data['unit_indices'][i] for i in filtered_indices]
                    units_data['r2_values'] = r2_values
                    units_data['filtered_indices'] = filtered_indices
                
                # Calculate preferred metrics
                if args.verbose:
                    print(f"    Calculating preferred metrics for {len(units_data['unit_rfs'])} units")
                
                metrics_df = calculate_all_metrics(
                    nwb_data=nwb_data,
                    units_data=units_data,
                    mouse_name=mouse_name,
                    probe=probe,
                    use_curvefit=args.curvefit,
                    verbose=args.verbose
                )
                
                if metrics_df is not None and not metrics_df.empty:
                    all_results.append(metrics_df)
                    print(f"    ✓ Successfully processed {len(metrics_df)} units")
                    
                    # Save raw RF data if requested
                    if args.save_raw and all_rf_data is not None:
                        for idx, rf in enumerate(units_data['unit_rfs']):
                            all_rf_data.append({
                                'mouse_name': mouse_name,
                                'unit_index': units_data['unit_indices'][idx],
                                'rf': rf
                            })
                else:
                    print(f"    Warning: No metrics calculated")
                
            except Exception as e:
                print(f"    Error processing {mouse_name}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Combine all results for this probe
        if not all_results:
            print(f"  Warning: No results were generated for {probe}")
            print()
            continue
        
        print()
        print("  " + "=" * 76)
        print(f"  Combining and saving results for {probe}...")
        print("  " + "=" * 76)
        
        final_df = pd.concat(all_results, ignore_index=True)
        print(f"  Total units analyzed: {len(final_df)}")
        print(f"  Mice processed: {final_df['mouse_name'].nunique()}")
        print()
        
        # Add to master results across all probes
        all_probes_results.append(final_df)
        
        # Save results
        save_results(
            df=final_df,
            output_dir=output_dir,
            rf_data=all_rf_data,
            args=args
        )
        
        # Generate plots
        if not args.no_plots:
            print(f"  Generating plots for {probe}...")
            plot_summary_figures(final_df, output_dir)
            plot_metric_distributions(final_df, output_dir)
            
            if args.filtered:
                plot_rf_examples(final_df, output_dir, n_examples=20)
            
            print(f"  ✓ Plots saved to {output_dir}")
        
        print()
    
    # Save master CSV across all probes and mice
    if all_probes_results:
        print("=" * 80)
        print("Saving master results across all probes...")
        print("=" * 80)
        
        master_df = pd.concat(all_probes_results, ignore_index=True)
        master_output_dir = Path(args.output_dir)
        master_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filter_str = f"filtered_r2-{args.r2_threshold:.2f}" if args.filtered else ("all_units" if args.all_units else "snr_filtered")
        master_filename = f"{timestamp}_all_probes_{filter_str}_results.csv"
        master_csv_path = master_output_dir / master_filename
        
        master_df.to_csv(master_csv_path, index=False)
        
        print(f"Master CSV saved: {master_csv_path}")
        print(f"  Total units: {len(master_df)}")
        print(f"  Total mice: {master_df['mouse_name'].nunique()}")
        print(f"  Probes: {', '.join(sorted(master_df['probe'].unique()))}")
        print()
    
    print("=" * 80)
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()