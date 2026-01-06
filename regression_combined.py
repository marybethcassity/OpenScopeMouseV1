import argparse
import numpy as np
import sys
from pathlib import Path
import pandas as pd

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
        fill in later"""
    )

     # Required arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to directory containing metric csv'
    )

def main():
    """Main execution function."""
    args = parse_arguments() 

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)

    discrete_variables = ['osi_dg', 'dsi_dg', 'probe']
    continuous_variables = ['pref_ori', 'pref_tf', 'pref_sf', 'snr']

    metrics_csv = list(Path(data_dir).glob("*metrics.csv"))
    
    if metrics_csv:
        print(f"Loading metrics from {metrics_csv[0]}")
    if not metrics_csv:
        print("Error: No metrics CSV file found in the specified directory.")
        sys.exit(1) 

    df = pd.read_csv(data_dir / metrics_csv[0])

    X = df[['rf_x_center', 'rf_y_center']].values

    results = {}

    for variable in discrete_variables:
        print(f"\nLogistic regression for discrete variable: {variable}")

        y = df[variable].values

        if variable == 'pref_sf':
                y = y * 100

        y = y.astype(str)

        counts = df[variable].value_counts()
        chance_proportional = (counts/len(df))**2
        chance_proportional_accuracy = chance_proportional.sum()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        model = LogisticRegression(
            multi_class = 'multinomial',
            solver = 'lbfgs',
            max_iter = 1000,
            random_state = 42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Chance proportional accuracy: {chance_proportional_accuracy:.4f}")

        results[variable] = {
                'accuracy': accuracy,
                'chance': chance_proportional_accuracy,
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'classes': model.classes_
        }

    for variable in continuous_variables:
        print(f"\nLinear regression for continuous variable: {variable}")

        y = df[variable].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"  R²: {r2:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")

        results[variable] = {
            'r²': r2,
            'mse': mse,
            'mae': mae,
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }

if __name__ == "__main__":
    main()