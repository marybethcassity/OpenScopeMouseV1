import argparse
import numpy as np
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

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
        help='Path to directory containing metric csv'
    )

    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments() 

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)

    continuous_variables = ['osi_dg', 'dsi_dg']
    discrete_variables = ['pref_ori', 'pref_tf', 'pref_sf']

    results = {}
    
    # Dictionary to store data for concatenation
    all_data = {
        'pref_tf': [], 
        'pref_ori': [], 
        'pref_sf': [], 
        'osi_dg': [],
        'dsi_dg': [],
        'rf_x': [], 
        'rf_y': []
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

            X = df[['rf_x_center', 'rf_y_center']].values

            results[probe] = {}
            
            # Collect data for concatenation - only collect valid (non-NaN) data
            for variable in discrete_variables:
                if variable in df.columns:
                    valid_mask = df[variable].notna() & df['rf_x_center'].notna() & df['rf_y_center'].notna()
                    all_data[variable].extend(df.loc[valid_mask, variable].values)
                    all_data['rf_x'].extend(df.loc[valid_mask, 'rf_x_center'].values)
                    all_data['rf_y'].extend(df.loc[valid_mask, 'rf_y_center'].values)
            
            for variable in continuous_variables:
                if variable in df.columns:
                    valid_mask = df[variable].notna() & df['rf_x_center'].notna() & df['rf_y_center'].notna()
                    all_data[variable].extend(df.loc[valid_mask, variable].values)
                    if variable == continuous_variables[0]:  # Only add RF centers once per probe
                        all_data['rf_x'].extend(df.loc[valid_mask, 'rf_x_center'].values)
                        all_data['rf_y'].extend(df.loc[valid_mask, 'rf_y_center'].values)

            # Process discrete variables
            for variable in discrete_variables:
                print(f"\nLogistic regression for discrete variable: {variable} (Probe: {probe})")

                if variable not in df.columns:
                    print(f"  Warning: {variable} not found in dataframe")
                    continue

                # Create a mask for valid (non-NaN) values
                valid_mask = df[variable].notna() & df['rf_x_center'].notna() & df['rf_y_center'].notna()
                
                if valid_mask.sum() == 0:
                    print(f"  Warning: No valid data for {variable}")
                    continue
                
                # Filter data to remove NaN values
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

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_valid)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
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

            # Process continuous variables
            for variable in continuous_variables:
                print(f"\nLinear regression for continuous variable: {variable} (Probe: {probe})")

                if variable not in df.columns:
                    print(f"  Warning: {variable} not found in dataframe")
                    continue

                # Create a mask for valid (non-NaN) values
                valid_mask = df[variable].notna() & df['rf_x_center'].notna() & df['rf_y_center'].notna()
                
                if valid_mask.sum() == 0:
                    print(f"  Warning: No valid data for {variable}")
                    continue
                
                # Filter data to remove NaN values
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
        n_samples = len(all_data[variable])
        X = np.column_stack([all_data['rf_x'][:n_samples], all_data['rf_y'][:n_samples]])
        y = np.array(all_data[variable])
        
        if variable == 'pref_sf':
            y = y * 100
        
        y = y.astype(str)
        
        # Calculate chance accuracy
        orientation_counts = pd.Series(y).value_counts()
        print(f"    Class counts: {dict(orientation_counts)}")
        
        chance_proportional = (orientation_counts/len(y))**2
        chance_accuracy_proportional = chance_proportional.sum()
        print(f"    Chance accuracy (proportional): {chance_accuracy_proportional:.4f}")
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
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
        n_samples = len(all_data[variable])
        X_all = np.column_stack([all_data['rf_x'][:n_samples], all_data['rf_y'][:n_samples]])
        y_all = np.array(all_data[variable])
        
        # Remove NaN values
        valid_mask = ~np.isnan(y_all) & ~np.isnan(X_all[:, 0]) & ~np.isnan(X_all[:, 1])
        
        if valid_mask.sum() == 0:
            print(f"    Warning: No valid data for {variable}")
            continue
        
        X = X_all[valid_mask]
        y = y_all[valid_mask]
        
        print(f"    Valid samples: {len(y)} out of {len(y_all)}")
        
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
    
    # Save to the parent data directory
    plt.savefig(os.path.join(data_dir, 'accuracy_discrete.png'), dpi=300, bbox_inches='tight')
    
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
    
    # Save to the parent data directory
    plt.savefig(os.path.join(data_dir, 'r2_scores_continuous.png'), dpi=300, bbox_inches='tight')
    
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

if __name__ == "__main__":
    main()