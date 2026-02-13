#!/usr/bin/env python3
"""
Main script for analyzing preferred metrics from NWB files.

Usage:
    python compute_pref_variables.py --data_dir /path/to/data 
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from pynwb import NWBHDF5IO

# Import custom modules
from data_processing import (
    process_units,
    calculate_all_metrics,
)

from gaussian_filtering import filter_rfs_by_gaussian_fit

from plotting import (
    plot_summary_figures,
    plot_metric_distributions,
    plot_preferred_orientation_bar,
    plot_preferred_tf_bar,
    plot_preferred_sf_bar
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
  python compute_pref_variables.py --data_dir /path/to/data --probe ProbeA --filtered
  
  # With custom R² threshold
  python compute_pref_variables.py --data_dir /path/to/data --probe ProbeA --filtered --r2_threshold 0.7
  
  # Without filtering
  python compute_pref_variables.py --data_dir /path/to/data --probe ProbeA --all_units
  
  # Only combine existing results (skip processing)
  python compute_pref_variables.py --data_dir /path/to/data --output_dir ../results/existing_results --combine_only
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to directory containing NWB files (organized by mouse)'
    )
    
    # Filtering options
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        '--filtered',
        action='store_true',
        help='Apply Gaussian fitting filter to receptive fields'
    )

    # filter_group.add_argument(
    #     '--all_units',
    #     action='store_true',
    #     help='Include all units without filtering'
    # )
    
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

    # Probe options 
    parser.add_argument(
        '--probe',
        type=str,
        default=None,
        help='Probe name to analyze (e.g., ProbeA, ProbeB). If not specified, analyzes all probes.'
    )

    # Combine only option
    parser.add_argument(
        '--combine_only',
        action='store_true',
        help='Skip NWB processing and only combine existing CSV results from output directory'
    )

    
    return parser.parse_args()

import ast

# TO DO: double check this-- see other TO DO below.
def combine_existing_results(output_dir, args):
    """
    Combine existing CSV results from a results directory.
    
    Parameters:
    -----------
    output_dir : Path
        Path to existing results directory
    args : argparse.Namespace
        Command line arguments
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("Combining existing results...")
    print("=" * 80)
    print(f"Searching for CSV files in: {output_dir}")
    print()
    
    # Find all CSV files in mouse/probe subdirectories
    all_csvs = []
    for mouse_dir in output_dir.iterdir():
        if not mouse_dir.is_dir() or not mouse_dir.name.startswith('sub-'):
            continue
        
        for probe_dir in mouse_dir.iterdir():
            if not probe_dir.is_dir():
                continue
            
            csv_files = list(probe_dir.glob("*_metrics.csv"))
            for csv_file in csv_files:
                all_csvs.append(csv_file)
                print(f"  Found: {csv_file.relative_to(output_dir)}")
    
    if not all_csvs:
        print("Error: No CSV files found in output directory")
        print("Expected structure: output_dir/sub-XXXXX/ProbeX/*_metrics.csv")
        sys.exit(1)
    
    print(f"\nFound {len(all_csvs)} CSV files")
    print()
    
    # Load and combine all CSVs
    all_dfs = []
    for csv_path in all_csvs:
        try:
            df = pd.read_csv(csv_path)
            
            # Convert RF column from string back to numpy array
            if 'rf' in df.columns:
                def parse_rf(rf_str):
                    try:
                        if pd.isna(rf_str):
                            return None
                        if isinstance(rf_str, str):
                            return np.array(ast.literal_eval(rf_str))
                        else:
                            return rf_str
                    except (ValueError, SyntaxError) as e:
                        print(f"    Warning: Could not parse RF in {csv_path.name}: {e}")
                        return None
                
                df['rf'] = df['rf'].apply(parse_rf)
            
            all_dfs.append(df)
            print(f"  ✓ Loaded: {csv_path.name} ({len(df)} units)")
            
        except Exception as e:
            print(f"  ✗ Warning: Could not load {csv_path.name}: {e}")
            continue
    
    if not all_dfs:
        print("Error: Could not load any CSV files")
        sys.exit(1)
    
    print(f"\n✓ Successfully loaded {len(all_dfs)} CSV files")
    print()
    
    # Combine all results
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save master CSV
    master_csv_path = output_dir / "all_mice_all_probes_results.csv"
    master_df.to_csv(master_csv_path, index=False)
    
    print(f"Master CSV saved: {master_csv_path}")
    print(f"  Total units: {len(master_df)}")
    print(f"  Total mice: {master_df['mouse_name'].nunique()}")
    print(f"  Total probes: {master_df['probe'].nunique()}")
    print(f"  Probes: {', '.join(sorted(master_df['probe'].unique()))}")
    print()
    
    # Generate master plots
    if not args.no_plots:
        print("Generating master plots (all mice, all probes)...")
        empty_units_data = {}
        plot_metric_distributions(master_df, empty_units_data, output_dir, probe_name="all_mice_all_probes")
        plot_summary_figures(master_df, empty_units_data, output_dir, probe_name="all_mice_all_probes")
        print(f"✓ Master plots saved to {output_dir}")
        print()
    
    # Create per-probe combined results
    print("=" * 80)
    print("Creating per-probe results across all mice...")
    print("=" * 80)
    
    for probe in sorted(master_df['probe'].unique()):
        print(f"\nProcessing {probe}...")
        
        probe_df = master_df[master_df['probe'] == probe].copy()
        
        probe_dir = output_dir / probe
        probe_dir.mkdir(parents=True, exist_ok=True)
        
        probe_csv_path = probe_dir / f"{probe}_all_mice_results.csv"
        probe_df.to_csv(probe_csv_path, index=False)
        print(f"  ✓ Saved CSV: {probe_csv_path}")
        print(f"    Units: {len(probe_df)}, Mice: {probe_df['mouse_name'].nunique()}")
        
        if not args.no_plots:
            print(f"  Generating plots for {probe} (all mice)...")
            empty_units_data = {}
            
            # Distribution plots
            plot_metric_distributions(probe_df, empty_units_data, probe_dir, probe_name=f"{probe}_all_mice")
            
            # Summary figures
            plot_summary_figures(probe_df, empty_units_data, probe_dir, probe_name=f"{probe}_all_mice")

            # Normalized bar plots (non-nested and nested variants)
            shared_normalized_kwargs = dict(
                peak_dff_min=1.0,
                output_dir=probe_dir,
                probe_name=probe,
                mouse_name=f"All Mice (n={probe_df['mouse_name'].nunique()})",
                normalize=True
            )

            plot_preferred_orientation_bar(probe_df, **shared_normalized_kwargs)

            bar_configs = [
                (plot_preferred_tf_bar, [{"nested": False}, {"nested": True}]),
                (plot_preferred_sf_bar, [{"nested": False}, {"nested": True}]),
            ]

            for plot_fn, variants in bar_configs:
                for variant_kwargs in variants:
                    plot_fn(probe_df, **shared_normalized_kwargs, **variant_kwargs)

            print(f"  ✓ Plots saved to {probe_dir}")
    
    print()
    print("=" * 80)
    print(f"Analysis complete! Results saved to: {output_dir}")
    print("=" * 80)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # If combine_only flag is set, just combine existing results
    if args.combine_only:
        output_dir = Path(args.output_dir)
        combine_existing_results(output_dir, args)
        return
    
    # Otherwise, proceed with normal processing
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
    
    # Create main output directory with arguments in the name
    from utils import create_results_directory
    main_output_dir = create_results_directory(args)
    
    print(f"Output directory: {main_output_dir}")
    print()
    
    # Initialize storage for all results across all mice and probes
    all_mouse_results = []
    
    # Process each mouse
    for mouse_idx, mouse_dir in enumerate(mouse_dirs, 1):
        mouse_name = mouse_dir.name
        print("=" * 80)
        print(f"Processing Mouse {mouse_idx}/{len(mouse_dirs)}: {mouse_name}")
        print("=" * 80)
        
        # Create mouse output directory
        mouse_output_dir = main_output_dir / mouse_name
        mouse_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find NWB file
        nwb_files = list(mouse_dir.glob("*.nwb"))
        if not nwb_files:
            print(f"  Warning: No NWB file found in {mouse_dir}")
            print()
            continue
        
        nwb_path = nwb_files[0]
        
        try:
            # Load NWB data once for this mouse
            if args.verbose:
                print(f"  Loading NWB file: {nwb_path.name}")
            
            io = NWBHDF5IO(str(nwb_path), 'r', load_namespaces=True)
            nwb = io.read()

            units = nwb.units
            rf_stim_table = nwb.intervals['receptive_field_block_presentations'].to_dataframe()
            dg_stim_table = nwb.intervals['drifting_gratings_field_block_presentations'].to_dataframe()

            nwb_data = {
                'nwb': nwb,
                'units': units, 
                'rf_stim_table': rf_stim_table,
                'dg_stim_table': dg_stim_table
            }

            # Determine which probes to analyze for this mouse
            if args.probe:
                probes_to_analyze = [args.probe]
                print(f"  Analyzing single probe: {args.probe}")
            else:
                # Auto-detect all probes from this mouse
                probes_to_analyze = sorted(set(units['device_name'][:]))
                print(f"  Found {len(probes_to_analyze)} probes: {', '.join(probes_to_analyze)}")
            
            print()
            
            # Store results for this mouse (all probes)
            mouse_results = []
            
            # Process each probe for this mouse
            for probe_idx, probe in enumerate(probes_to_analyze, 1):
                print(f"  {'-' * 76}")
                print(f"  Processing Probe {probe_idx}/{len(probes_to_analyze)}: {probe}")
                print(f"  {'-' * 76}")
                
                # Create probe-specific subdirectory under mouse
                probe_output_dir = mouse_output_dir / probe
                probe_output_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"    Probe: {probe}")
                print(f"    Filtering: {'Gaussian fit (R² >= ' + str(args.r2_threshold) + ')' if args.filtered else 'All units'}")
                print(f"    Output directory: {probe_output_dir}")
                print()
                
                try:
                    # Process units and calculate receptive fields
                    if args.verbose:
                        print(f"    Processing units for probe: {probe}")
                    
                    units_data = process_units(
                        nwb_data=nwb_data,
                        probe=probe,
                        all_units=not args.filtered,
                        verbose=args.verbose
                    )
                    
                    if units_data['unit_rfs'] is None or len(units_data['unit_rfs']) == 0:
                        print(f"    No units found for {probe}")
                        print()
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
                        mouse_results.append(metrics_df)
                        print(f"    ✓ Successfully processed {len(metrics_df)} units")
                        
                        # Save probe-specific CSV
                        probe_csv_path = probe_output_dir / f"{probe}_metrics.csv"
                        metrics_df.to_csv(probe_csv_path, index=False)
                        print(f"    ✓ Saved probe results to {probe_csv_path}")
                        
                        # Generate plots for this probe
                        if not args.no_plots:
                            print(f"    Generating plots for {probe}...")
                            plot_metric_distributions(metrics_df, units_data, probe_output_dir, probe_name=probe)
                            plot_summary_figures(metrics_df, units_data, probe_output_dir, probe_name=probe)
                            
                            print(f"    ✓ Plots saved to {probe_output_dir}")
                    else:
                        print(f"    Warning: No metrics calculated")
                    
                except Exception as e:
                    print(f"    Error processing probe {probe}: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    continue
                
                print()
            
            # Store results for this mouse
            if mouse_results:
                print(f"  ✓ Processed {len(mouse_results)} probe(s) for {mouse_name}")
                
                # Add to master results
                for probe_df in mouse_results:
                    all_mouse_results.append(probe_df)
                print()

                # Generate master plots across all mice and all probes
                if not args.no_plots:
                    print("Generating master plots for mouse...")
                    empty_units_data = {}
                    combined_mouse_results = pd.concat(mouse_results, ignore_index=True)
                    plot_metric_distributions(combined_mouse_results, empty_units_data, mouse_output_dir, probe_name="all_probes")
                    plot_summary_figures(combined_mouse_results, empty_units_data, mouse_output_dir, probe_name="all_probes")
                    print(f"✓ Master plots saved to {mouse_output_dir}")
                    print()

            else:
                print(f"  Warning: No results generated for {mouse_name}")
                print()
                
        except Exception as e:
            print(f"  Error processing mouse {mouse_name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            print()
            continue
        
        io.close()

    # Combine and save results
    if all_mouse_results:
        print("=" * 80)
        print("Combining results across all mice...")
        print("=" * 80)
        
        # Combine all results
        master_df = pd.concat(all_mouse_results, ignore_index=True)
        
        # Save master CSV (all mice, all probes)
        master_filename = "all_mice_all_probes_results.csv"
        master_csv_path = main_output_dir / master_filename
        master_df.to_csv(master_csv_path, index=False)
        
        print(f"\nMaster CSV saved: {master_csv_path}")
        print(f"  Total units: {len(master_df)}")
        print(f"  Total mice: {master_df['mouse_name'].nunique()}")
        print(f"  Total probes: {master_df['probe'].nunique()}")
        print(f"  Probes: {', '.join(sorted(master_df['probe'].unique()))}")
        print()
        
        # Generate master plots across all mice and all probes
        if not args.no_plots:
            print("Generating master plots (all mice, all probes)...")
            empty_units_data = {}
            plot_metric_distributions(master_df, empty_units_data, main_output_dir, probe_name="all_mice_all_probes")
            plot_summary_figures(master_df, empty_units_data, main_output_dir, probe_name="all_mice_all_probes")
            print(f"✓ Master plots saved to {main_output_dir}")
            print()
        
        # Create per-probe combined results across all mice
        print("=" * 80)
        print("Creating per-probe results across all mice...")
        print("=" * 80)
        
        for probe in sorted(master_df['probe'].unique()):
            print(f"\nProcessing {probe}...")
            
            # Filter data for this probe
            probe_df = master_df[master_df['probe'] == probe].copy()
            
            # Create probe directory in main output
            probe_dir = main_output_dir / probe
            probe_dir.mkdir(parents=True, exist_ok=True)
            
            # Save probe-specific CSV across all mice
            probe_csv_path = probe_dir / f"{probe}_all_mice_results.csv"
            probe_df.to_csv(probe_csv_path, index=False)
            print(f"  ✓ Saved CSV: {probe_csv_path}")
            print(f"    Units: {len(probe_df)}, Mice: {probe_df['mouse_name'].nunique()}")
            
            # Generate plots for this probe across all mice
            if not args.no_plots:
                print(f"  Generating plots for {probe} (all mice)...")
                empty_units_data = {}
                
                # TO DO: this doesn't make a ton of sense how I did it-- we don't really need the non normalized 
                # version of the bar plots so I should really only be making them once while calling plot_metrics_distributions and plot_summary_figures, but I want to get the normalized ones in there for now so I'm just calling them separately for now. I can clean this up later.
                # instead of twice by calling the bar plots again and passing in normalize = True

                # Distribution plots
                plot_metric_distributions(probe_df, empty_units_data, probe_dir, probe_name=f"{probe}_all_mice")
                
                # Summary figures
                plot_summary_figures(probe_df, empty_units_data, probe_dir, probe_name=f"{probe}_all_mice")
                
                # Normalized bar plots (non-nested and nested variants)
                shared_normalized_kwargs = dict(
                    peak_dff_min=1.0,
                    output_dir=probe_dir,
                    probe_name=probe,
                    mouse_name=f"All Mice (n={probe_df['mouse_name'].nunique()})",
                    normalize=True
                )

                plot_preferred_orientation_bar(probe_df, **shared_normalized_kwargs)

                bar_configs = [
                    (plot_preferred_tf_bar, [{"nested": False}, {"nested": True}]),
                    (plot_preferred_sf_bar, [{"nested": False}, {"nested": True}]),
                ]

                for plot_fn, variants in bar_configs:
                    for variant_kwargs in variants:
                        plot_fn(probe_df, **shared_normalized_kwargs, **variant_kwargs)
                
                print(f"  ✓ Plots saved to {probe_dir}")
            
            print()
        
        print("=" * 80)
        print(f"Analysis complete! Results saved to: {main_output_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()