import argparse
import numpy as np
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Analyze preferred metrics from NWB files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example usage:
        python regression_by_mouse.py --data_dir /path/to/data"""
    )

    # Required arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to directory containing metric csv. Ex: "X:\Personnel\MaryBeth\OpenScope\001568\results\results_all-probes_filtered-r2-0.50_20260108_144341"'
    )

    # Optional arguments
    parser.add_argument(
        '--filtering',
        type=str,
        default=None,
        help='Filtering method to apply (e.g., "peak_prominence")'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.25,
        help='Threshold value for filtering (e.g., 0.25)'
    )

    return parser.parse_args()

def calculate_peak_prominence(responses_str):
    """
    Calculate peak prominence from response string.
    
    Args:
        responses_str: String representation of response array
        
    Returns:
        Peak prominence value or NaN if invalid
    """
    try:
        # Parse the string representation of the array
        responses = np.array(eval(responses_str))
        max_resp = np.max(responses)
        mean_resp = np.mean(responses)
        peak_prominence = (max_resp - mean_resp) / (max_resp + 1e-6)
        return peak_prominence
    except:
        return np.nan

def apply_peak_prominence_calc(df):
    """
    Calculate peak prominence for all response types and add as columns.
    
    Args:
        df: DataFrame with response columns
        
    Returns:
        DataFrame with prominence columns added and prominence statistics dictionary
    """
    prominence_stats = {
        'ori': {'values': [], 'mean': np.nan, 'median': np.nan, 'std': np.nan},
        'tf': {'values': [], 'mean': np.nan, 'median': np.nan, 'std': np.nan},
        'sf': {'values': [], 'mean': np.nan, 'median': np.nan, 'std': np.nan}
    }
    
    # Calculate peak prominence for each response type
    if 'ori_responses' in df.columns:
        df['ori_peak_prominence'] = df['ori_responses'].apply(calculate_peak_prominence)
        valid_ori = df['ori_peak_prominence'].dropna()
        if len(valid_ori) > 0:
            prominence_stats['ori']['values'] = valid_ori.values
            prominence_stats['ori']['mean'] = valid_ori.mean()
            prominence_stats['ori']['median'] = valid_ori.median()
            prominence_stats['ori']['std'] = valid_ori.std()
    
    if 'tf_responses' in df.columns:
        df['tf_peak_prominence'] = df['tf_responses'].apply(calculate_peak_prominence)
        valid_tf = df['tf_peak_prominence'].dropna()
        if len(valid_tf) > 0:
            prominence_stats['tf']['values'] = valid_tf.values
            prominence_stats['tf']['mean'] = valid_tf.mean()
            prominence_stats['tf']['median'] = valid_tf.median()
            prominence_stats['tf']['std'] = valid_tf.std()
    
    if 'sf_responses' in df.columns:
        df['sf_peak_prominence'] = df['sf_responses'].apply(calculate_peak_prominence)
        valid_sf = df['sf_peak_prominence'].dropna()
        if len(valid_sf) > 0:
            prominence_stats['sf']['values'] = valid_sf.values
            prominence_stats['sf']['mean'] = valid_sf.mean()
            prominence_stats['sf']['median'] = valid_sf.median()
            prominence_stats['sf']['std'] = valid_sf.std()
    
    # Don't filter here - just return df with prominence columns added
    return df, prominence_stats

def main():
    """Main execution function."""
    args = parse_arguments() 

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)

    # Create output directory with timestamp and filtering argument
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filtering_str = args.filtering if args.filtering else "no_filter"
    threshold = args.threshold if args.filtering else "N/A"

    if args.filtering:
        output_dir_name = f"regression_{timestamp}_{filtering_str}_{threshold}"
    else:
        output_dir_name = f"regression_{timestamp}_no_filter"  

    output_dir = data_dir / "regression" / output_dir_name
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nResults will be saved to: {output_dir}")

    continuous_variables = ['osi_dg', 'dsi_dg']
    discrete_variables = ['pref_ori', 'pref_tf', 'pref_sf']

    results = {}
    
    # Dictionary to store prominence statistics across all probes
    all_prominence_stats = {
        'ori': [],
        'tf': [],
        'sf': []
    }
    
    # List to track skipped probes due to insufficient data
    skipped_probes = []
    
    # Dictionary to store data for concatenation
    all_data = {
        'pref_tf': pd.DataFrame(columns=['variable', 'rf_x', 'rf_y']),
        'pref_ori': pd.DataFrame(columns=['variable', 'rf_x', 'rf_y']),
        'pref_sf': pd.DataFrame(columns=['variable', 'rf_x', 'rf_y']),
        'osi_dg': pd.DataFrame(columns=['variable', 'rf_x', 'rf_y']),
        'dsi_dg': pd.DataFrame(columns=['variable', 'rf_x', 'rf_y'])
    }

    for mouse_dir in data_dir.iterdir():
        if not mouse_dir.is_dir() or not mouse_dir.name.startswith('sub-'):
            continue
        
        for probe_dir in mouse_dir.iterdir():
            if not probe_dir.is_dir():
                continue
            
            probe = probe_dir.name

            # Look for CSV files with metrics
            metrics_csv = list(probe_dir.glob("*_metrics.csv"))
    
            if metrics_csv:
                print(f"\nLoading metrics from {metrics_csv[0]}")
            if not metrics_csv:
                print(f"Warning: No metrics CSV file found in {probe_dir}")
                continue

            df = pd.read_csv(metrics_csv[0])

            # Apply peak prominence filtering if requested
            if args.filtering == 'peak_prominence':
                df, prominence_stats = apply_peak_prominence_calc(df)
                
                # Collect prominence values for later plotting
                if prominence_stats['ori']['values'] is not None and len(prominence_stats['ori']['values']) > 0:
                    all_prominence_stats['ori'].extend(prominence_stats['ori']['values'])
                if prominence_stats['tf']['values'] is not None and len(prominence_stats['tf']['values']) > 0:
                    all_prominence_stats['tf'].extend(prominence_stats['tf']['values'])
                if prominence_stats['sf']['values'] is not None and len(prominence_stats['sf']['values']) > 0:
                    all_prominence_stats['sf'].extend(prominence_stats['sf']['values'])
            
            X = df[['rf_x_center', 'rf_y_center']].values

            results[probe] = {}

            # Process discrete variables
            probe_has_sufficient_data = True  # Track if this probe should be skipped
            skip_reasons = []  # Track which variables caused skipping
            
            # Collect data for concatenation - only collect valid (non-NaN) data
            # Apply variable-specific prominence filtering when collecting data
            for variable in discrete_variables:
                if variable in df.columns:
                    valid_mask = df[variable].notna() & df['rf_x_center'].notna() & df['rf_y_center'].notna()
                    
                    # Apply variable-specific prominence filter if enabled
                    if args.filtering == 'peak_prominence':
                        prominence_col_map = {
                            'pref_ori': 'ori_peak_prominence',
                            'pref_tf': 'tf_peak_prominence',
                            'pref_sf': 'sf_peak_prominence'
                        }
                        if variable in prominence_col_map:
                            prominence_col = prominence_col_map[variable]
                            if prominence_col in df.columns:
                                valid_mask = valid_mask & (df[prominence_col] >= threshold)
                    
                    temp_df = pd.DataFrame({
                        'variable': df.loc[valid_mask, variable].values,
                        'rf_x': df.loc[valid_mask, 'rf_x_center'].values,
                        'rf_y': df.loc[valid_mask, 'rf_y_center'].values
                    })
        
                    all_data[variable] = pd.concat([all_data[variable], temp_df], ignore_index=True)

                # Get data for regression
                X_valid = df.loc[valid_mask, ['rf_x_center', 'rf_y_center']].values
                y = df.loc[valid_mask, variable].values



                if variable == 'pref_sf':
                    y = y * 100

                y = y.astype(str)

                # Check if we have enough classes for stratification
                unique_classes = np.unique(y)
                if len(unique_classes) < 2:
                    print(f"  Warning: Only {len(unique_classes)} unique class(es) found. Skipping.")
                    continue

                counts = pd.Series(y).value_counts()
                chance_proportional = (counts/len(y))**2
                chance_proportional_accuracy = chance_proportional.sum()

                print(f"  Valid samples: {len(y)} out of {len(df)}")
                print(f"  Class distribution: {dict(counts)}")

                # Check if any class has fewer than 2 samples (required for stratified split)
                min_class_count = counts.min()
                if min_class_count < 2:
                    print(f"  Warning: Least populated class has only {min_class_count} sample(s). Need at least 2 per class.")
                    print(f"  SKIPPING ENTIRE PROBE: {probe}")
                    probe_has_sufficient_data = False
                    skip_reasons.append(f"{variable}: {len(y)} samples, least populated class has {min_class_count}")
                    break  # Skip this entire probe

                # Adjust test_size for small datasets to ensure enough samples per class
                # Rule: test_size needs at least 1 sample per class, ideally 2+
                min_test_samples = len(unique_classes) * 2  # At least 2 samples per class in test
                
                if len(y) < min_test_samples + len(unique_classes):
                    print(f"  Warning: Dataset too small ({len(y)} samples, {len(unique_classes)} classes) for reliable train/test split.")
                    print(f"  SKIPPING ENTIRE PROBE: {probe}")
                    probe_has_sufficient_data = False
                    skip_reasons.append(f"{variable}: {len(y)} samples, {len(unique_classes)} classes")
                    break  # Skip this entire probe
                
                # Calculate appropriate test_size
                if len(y) < 50:
                    # For very small datasets, use larger test fraction but ensure minimum samples
                    test_size = max(0.3, min_test_samples / len(y))
                else:
                    test_size = 0.2
                
                # Ensure test_size doesn't exceed 0.5
                test_size = min(test_size, 0.5)

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_valid)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42, stratify=y
                )

                model = LogisticRegression(
                    multi_class='multinomial',
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Chance proportional accuracy: {chance_proportional_accuracy:.4f}")

                results[probe][variable] = {
                    'accuracy': accuracy,
                    'chance': chance_proportional_accuracy,
                    'coefficients': model.coef_,
                    'intercept': model.intercept_,
                    'classes': model.classes_
                }
            
            # If probe should be skipped, record it and skip continuous variables too
            if not probe_has_sufficient_data:
                mouse_name = mouse_dir.name
                skipped_probes.append({
                    'mouse': mouse_name,
                    'probe': probe,
                    'reason': '; '.join(skip_reasons)
                })
                print(f"\n*** PROBE {probe} SKIPPED - Insufficient data after filtering ***\n")
                continue  # Skip to next probe
            
            for variable in continuous_variables:
                if variable in df.columns:
                    valid_mask = df[variable].notna() & df['rf_x_center'].notna() & df['rf_y_center'].notna()
                    
                    # Apply variable-specific prominence filter if enabled
                    if args.filtering == 'peak_prominence':
                        prominence_col_map = {
                            'osi_dg': 'ori_peak_prominence',
                            'dsi_dg': 'ori_peak_prominence'
                        }
                        if variable in prominence_col_map:
                            prominence_col = prominence_col_map[variable]
                            if prominence_col in df.columns:
                                valid_mask = valid_mask & (df[prominence_col] >= 0.25)
                    
                    temp_df = pd.DataFrame({
                        'variable': df.loc[valid_mask, variable].values,
                        'rf_x': df.loc[valid_mask, 'rf_x_center'].values,
                        'rf_y': df.loc[valid_mask, 'rf_y_center'].values
                    })
                    
                    all_data[variable] = pd.concat([all_data[variable], temp_df], ignore_index=True)
                    
                # Get data for regression
                X_valid = df.loc[valid_mask, ['rf_x_center', 'rf_y_center']].values
                y = df.loc[valid_mask, variable].values
                
                print(f"  Valid samples: {len(y)} out of {len(df)}")

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_valid)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )

                model = LinearRegression()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                print(f"  R²: {r2:.4f}")
                print(f"  MSE: {mse:.4f}")
                print(f"  MAE: {mae:.4f}")

                results[probe][variable] = {
                    'r2': r2,
                    'mse': mse,
                    'mae': mae,
                    'coefficients': model.coef_,
                    'intercept': model.intercept_
                }

    # Process concatenated data (All)
    print("\n" + "="*80)
    print("Processing: ALL_CONCATENATED")
    print("="*80)
    results['All'] = {}

    # Process discrete variables for concatenated data
    for variable in discrete_variables:
        if len(all_data[variable]) == 0:
            continue
        
        print(f"\n  {variable}:")
        
        # Prepare concatenated data
        X = all_data[variable][['rf_x', 'rf_y']].values
        y = all_data[variable]['variable'].values

        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        if n_classes <= 10:
            class_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        else:
            class_colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

        x_positions = X[:, 0]
        y_positions = X[:, 1]

        # # Get RF extent from positions
        # x_min, x_max = -40, 40
        # y_min, y_max = -40, 40

        # # Add padding
        # x_padding = (x_max - x_min) * 0.1
        # y_padding = (y_max - y_min) * 0.1

        fig, ax = plt.subplots(figsize=(12, 10))
        
        for i, class_ in enumerate(unique_classes):
            class_mask = (y == class_)
            ax.scatter(x_positions[class_mask], y_positions[class_mask], color=class_colors[i], label=f'{class_}', s=100, alpha=0.6,  
                       edgecolors='black', linewidths=1.0)
        
        ax.set_xlabel('RF Center X Position (degrees)', fontsize=14)
        ax.set_ylabel('RF Center Y Position (degrees)', fontsize=14)

        title = f'{variable} distribution.png'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_dir / title, dpi=300, bbox_inches='tight')
        
        if variable == 'pref_sf':
            y = y * 100
        
        y = y.astype(str)
        
        # Calculate chance accuracy
        orientation_counts = pd.Series(y).value_counts()
        print(f"    Class counts: {dict(orientation_counts)}")
        
        chance_proportional = (orientation_counts/len(y))**2
        chance_accuracy_proportional = chance_proportional.sum()
        print(f"    Chance accuracy (proportional): {chance_accuracy_proportional:.4f}")
        
        # Check if any class has fewer than 2 samples (required for stratified split)
        min_class_count = orientation_counts.min()
        if min_class_count < 2:
            print(f"    Warning: Least populated class has only {min_class_count} sample(s). Need at least 2 per class. Skipping.")
            continue
        
        # Check if dataset is large enough for train/test split
        unique_classes = np.unique(y)
        min_test_samples = len(unique_classes) * 2
        
        if len(y) < min_test_samples + len(unique_classes):
            print(f"    Warning: Dataset too small ({len(y)} samples, {len(unique_classes)} classes) for reliable train/test split. Skipping.")
            continue
        
        # Calculate appropriate test_size
        if len(y) < 50:
            test_size = max(0.3, min_test_samples / len(y))
        else:
            test_size = 0.2
        
        test_size = min(test_size, 0.5)
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"    Model accuracy: {accuracy:.4f}")
        
        # Store results
        results['All'][variable] = {
            'accuracy': accuracy,
            'chance': chance_accuracy_proportional,
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'classes': model.classes_
        }

    # Process continuous variables for concatenated data
    for variable in continuous_variables:
        if len(all_data[variable]) == 0:
            continue
        
        print(f"\n  {variable}:")
        
        # Prepare concatenated data
        X = all_data[variable][['rf_x', 'rf_y']].values
        y = all_data[variable]['variable'].values
        
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() == 0:
            print(f"    Warning: No valid data for {variable}")
            continue
        
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"    R²: {r2:.4f}")
        print(f"    MSE: {mse:.4f}")
        print(f"    MAE: {mae:.4f}")
        
        # Store results
        results['All'][variable] = {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }

    print()
    print("="*80)

    # Create the bar graph for discrete variables
    probes = sorted([p for p in results.keys() if p != 'All'])
    probes.append('All')  # Add concatenated at the end

    variables = ['pref_tf', 'pref_ori', 'pref_sf']

    # Colors for each variable
    variable_colors = {
        'pref_tf': 'darkorange',
        'pref_ori': 'mediumorchid',
        'pref_sf': 'orangered'
    }

    fig, ax = plt.subplots(figsize=(16, 8))

    bar_width = 0.8 / len(variables)
    x = np.arange(len(probes))

    for var_idx, variable in enumerate(variables):
        positions = x + var_idx * bar_width
        
        accuracies = []
        chances = []
        
        for probe in probes:
            if variable in results[probe]:
                accuracies.append(results[probe][variable]['accuracy'])
                chances.append(results[probe][variable]['chance'])
            else:
                accuracies.append(0)
                chances.append(0)
        
        color = variable_colors[variable]
        
        # Bars with white fill and colored edge for accuracy 
        ax.bar(positions, accuracies, bar_width, 
            label=f'{variable}', 
            color='white', 
            edgecolor=color, 
            linewidth=2)
        
        # Bars with colored fill for chance accuracy
        ax.bar(positions, chances, bar_width, 
            color=color, 
            alpha=0.7)
        
        # Add percent difference labels above bars
        for i, (pos, acc, chance) in enumerate(zip(positions, accuracies, chances)):
            if acc > 0 and chance > 0:  # Only add label if both values exist
                percent_diff = ((acc - chance) / chance) * 100
                # Position label slightly above the accuracy bar
                ax.text(pos, acc + 0.02, f'+{percent_diff:.1f}%', 
                    ha='center', va='bottom', fontsize=8, 
                    color=color, fontweight='bold')

    ax.set_xlabel('Probe', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')

    ax.set_title('Model Accuracy vs Chance Accuracy by Probe (Discrete Variables)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(probes, rotation=45, ha='right')
    ax.legend(title='Variable', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)  # Increased ylim to accommodate labels

    plt.tight_layout()
    
    # Save to the output directory
    plt.savefig(output_dir / 'accuracy_discrete.png', dpi=300, bbox_inches='tight')
    
    plt.show()

    print("\nDiscrete variables plot saved!")

    # Print summary for discrete variables
    print("\nSummary (Discrete Variables):")
    for probe in probes:
        print(f"\n{probe}:")
        if 'pref_tf' in results[probe]:
            print(f"  pref_tf - Accuracy: {results[probe]['pref_tf']['accuracy']:.4f}, Chance: {results[probe]['pref_tf']['chance']:.4f}")
        if 'pref_ori' in results[probe]:
            print(f"  pref_ori - Accuracy: {results[probe]['pref_ori']['accuracy']:.4f}, Chance: {results[probe]['pref_ori']['chance']:.4f}")
        if 'pref_sf' in results[probe]:
            print(f"  pref_sf - Accuracy: {results[probe]['pref_sf']['accuracy']:.4f}, Chance: {results[probe]['pref_sf']['chance']:.4f}")

    # Create the bar graph for continuous variables
    variables_cont = ['osi_dg', 'dsi_dg']

    # Colors for each variable
    variable_colors_cont = {
        'osi_dg': 'blue',
        'dsi_dg': 'green'
    }

    fig, ax = plt.subplots(figsize=(16, 8))

    bar_width = 0.8 / len(variables_cont)
    x = np.arange(len(probes))

    for var_idx, variable in enumerate(variables_cont):
        positions = x + var_idx * bar_width
        
        r2_scores = []
        
        for probe in probes:
            if variable in results[probe]:
                r2_scores.append(results[probe][variable]['r2'])
            else:
                r2_scores.append(0)
        
        color = variable_colors_cont[variable]
        
        # Bars with colored fill for R²
        ax.bar(positions, r2_scores, bar_width, 
            label=f'{variable}', 
            color=color, 
            alpha=0.7)
        
        # Add R² value labels above bars
        for i, (pos, r2) in enumerate(zip(positions, r2_scores)):
            if r2 != 0:  # Only add label if R² value exists
                ax.text(pos, r2 + 0.02, f'R²={r2:.3f}', 
                    ha='center', va='bottom', fontsize=8, 
                    color=color, fontweight='bold')

    ax.set_xlabel('Probe', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')

    ax.set_title('Model R² by Probe (Continuous Variables)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(probes, rotation=45, ha='right')
    ax.legend(title='Variable', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Add a horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Set fixed y-limits
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    
    # Save to the output directory
    plt.savefig(output_dir / 'r2_scores_continuous.png', dpi=300, bbox_inches='tight')
    
    plt.show()

    print("\nContinuous variables plot saved!")

    # Print summary for continuous variables
    print("\nSummary (Continuous Variables):")
    for probe in probes:
        print(f"\n{probe}:")
        if 'osi_dg' in results[probe]:
            print(f"  osi_dg - R²: {results[probe]['osi_dg']['r2']:.4f}, MSE: {results[probe]['osi_dg']['mse']:.4f}")
        if 'dsi_dg' in results[probe]:
            print(f"  dsi_dg - R²: {results[probe]['dsi_dg']['r2']:.4f}, MSE: {results[probe]['dsi_dg']['mse']:.4f}")

    # Print linear regression coefficients
    print("\n" + "="*80)
    print("Linear Regression Coefficients:")
    print("="*80)
    print(f"{'Probe':<18} {'Variable':<10} {'RF_X_coef':<12} {'RF_Y_coef':<12} {'Intercept':<12}")
    print("-"*64)
    for probe in probes:
        for variable in variables_cont:
            if variable in results[probe]:
                coefs = results[probe][variable]['coefficients']
                intercept = results[probe][variable]['intercept']
                print(f"{probe:<18} {variable:<10} {coefs[0]:>11.4f} {coefs[1]:>11.4f} {intercept:>11.4f}")

    print(f"\nAll results saved to: {output_dir}")

    # Save skipped probes to CSV file
    if skipped_probes:
        skipped_df = pd.DataFrame(skipped_probes)
        skipped_csv_path = output_dir / 'skipped_probes.csv'
        skipped_df.to_csv(skipped_csv_path, index=False)
        print(f"\nSkipped probes saved to: {skipped_csv_path}")
        print(f"Total probes skipped: {len(skipped_probes)}")
    else:
        print("\nNo probes were skipped.")

    # Create box plots for peak prominence if filtering was applied
    if args.filtering == 'peak_prominence':
        print("\nCreating peak prominence statistics plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        response_types = ['ori', 'tf', 'sf']
        colors = ['steelblue', 'darkorange', 'forestgreen']
        
        for idx, (resp_type, color) in enumerate(zip(response_types, colors)):
            ax = axes[idx]
            
            if len(all_prominence_stats[resp_type]) > 0:
                data = all_prominence_stats[resp_type]
                
                # Create box plot
                bp = ax.boxplot([data], widths=0.6, patch_artist=True,
                               boxprops=dict(facecolor=color, alpha=0.7),
                               medianprops=dict(color='red', linewidth=2),
                               whiskerprops=dict(color=color),
                               capprops=dict(color=color))
                
                # Add statistics text
                mean_val = np.mean(data)
                median_val = np.median(data)
                std_val = np.std(data)
                min_val = np.min(data)
                max_val = np.max(data)
                
                stats_text = f'Mean: {mean_val:.3f}\n'
                stats_text += f'Median: {median_val:.3f}\n'
                stats_text += f'Std: {std_val:.3f}\n'
                stats_text += f'Min: {min_val:.3f}\n'
                stats_text += f'Max: {max_val:.3f}\n'
                stats_text += f'N: {len(data)}'
                
                ax.text(0.98, 0.98, stats_text,
                       transform=ax.transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9)
                
                # Add threshold line
                ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, 
                          label=f'Threshold ({threshold})', alpha=0.7)
                
                ax.set_ylabel('Peak Prominence', fontsize=11, fontweight='bold')
                ax.set_title(f'{resp_type.upper()} Responses', fontsize=12, fontweight='bold')
                ax.set_xticks([])
                ax.grid(axis='y', alpha=0.3)
                ax.legend(loc='lower right', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No Data', 
                       transform=ax.transAxes,
                       horizontalalignment='center',
                       verticalalignment='center',
                       fontsize=14)
                ax.set_title(f'{resp_type.upper()} Responses', fontsize=12, fontweight='bold')
        
        plt.suptitle('Peak Prominence Distribution (All Probes)', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / 'peak_prominence_stats.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Peak prominence statistics plot saved!")
        
        # Also create a histogram for each response type
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (resp_type, color) in enumerate(zip(response_types, colors)):
            ax = axes[idx]
            
            if len(all_prominence_stats[resp_type]) > 0:
                data = all_prominence_stats[resp_type]
                
                # Create histogram
                ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor='black')
                
                # Add threshold line
                ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                          label=f'Threshold ({threshold})', alpha=0.8)
                
                # Count cells above threshold
                n_above = np.sum(np.array(data) >= threshold)
                n_total = len(data)
                pct_above = (n_above / n_total) * 100
                
                ax.text(0.98, 0.98, f'{n_above}/{n_total} ({pct_above:.1f}%)\nabove threshold',
                       transform=ax.transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10)
                
                ax.set_xlabel('Peak Prominence', fontsize=11, fontweight='bold')
                ax.set_ylabel('Count', fontsize=11, fontweight='bold')
                ax.set_title(f'{resp_type.upper()} Responses', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                ax.legend(loc='upper left', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No Data', 
                       transform=ax.transAxes,
                       horizontalalignment='center',
                       verticalalignment='center',
                       fontsize=14)
                ax.set_title(f'{resp_type.upper()} Responses', fontsize=12, fontweight='bold')
        
        plt.suptitle('Peak Prominence Histogram (All Probes)', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / 'peak_prominence_histogram.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Peak prominence histogram saved!")

if __name__ == "__main__":
    main()