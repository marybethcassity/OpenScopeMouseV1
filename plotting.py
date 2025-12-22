"""
Plotting module for visualizing analysis results.
Includes summary figures, distributions, and receptive field examples.
Supports both per-mouse and combined plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_metric_distributions(df, output_dir):
    """
    Create distribution plots for each mouse individually AND combined.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe with columns: mouse_name, osi, dsi, pref_ori, pref_tf, pref_sf
    output_dir : Path
        Output directory
    """
    # Get list of unique mice
    mice = sorted(df['mouse_name'].unique())
    
    # Plot for each individual mouse
    for mouse in mice:
        mouse_df = df[df['mouse_name'] == mouse]
        
        # Create subdirectory for this mouse
        mouse_dir = output_dir / mouse
        mouse_dir.mkdir(exist_ok=True)
        
        # Plot all metrics for this mouse
        _plot_osi(mouse_df, mouse_dir, mouse_name=mouse)
        _plot_dsi(mouse_df, mouse_dir, mouse_name=mouse)
        _plot_pref_ori(mouse_df, mouse_dir, mouse_name=mouse)
        _plot_pref_tf(mouse_df, mouse_dir, mouse_name=mouse)
        _plot_pref_sf(mouse_df, mouse_dir, mouse_name=mouse)
    
    # Plot combined across all mice
    _plot_osi(df, output_dir, mouse_name='Combined')
    _plot_dsi(df, output_dir, mouse_name='Combined')
    _plot_pref_ori(df, output_dir, mouse_name='Combined')
    _plot_pref_tf(df, output_dir, mouse_name='Combined')
    _plot_pref_sf(df, output_dir, mouse_name='Combined')


def _plot_osi(df, output_dir, mouse_name=None):
    """Plot OSI distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    osis = df['osi'].dropna().values
    
    if len(osis) == 0:
        plt.close()
        return
    
    ax.hist(osis, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Orientation Selectivity Index (OSI)', fontsize=12)
    ax.set_ylabel('Number of Cells', fontsize=12)
    
    title = f'Orientation Selectivity Distribution\n({len(osis)} cells)'
    if mouse_name:
        title = f'{mouse_name}\n' + title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    stats_text = f'Mean: {np.mean(osis):.3f}\nMedian: {np.median(osis):.3f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    filename = 'combined_osi_distribution.png' if mouse_name == 'Combined' else 'osi_distribution.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_dsi(df, output_dir, mouse_name=None):
    """Plot DSI distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    dsis = df['dsi'].dropna().values
    
    if len(dsis) == 0:
        plt.close()
        return
    
    ax.hist(dsis, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Direction Selectivity Index (DSI)', fontsize=12)
    ax.set_ylabel('Number of Cells', fontsize=12)
    
    title = f'Direction Selectivity Distribution\n({len(dsis)} cells)'
    if mouse_name:
        title = f'{mouse_name}\n' + title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    stats_text = f'Mean: {np.mean(dsis):.3f}\nMedian: {np.median(dsis):.3f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    filename = 'combined_dsi_distribution.png' if mouse_name == 'Combined' else 'dsi_distribution.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_pref_ori(df, output_dir, mouse_name=None):
    """Plot preferred orientation distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    pref_oris = df['pref_ori'].dropna().values
    
    if len(pref_oris) == 0:
        plt.close()
        return
    
    unique_oris = np.sort(np.unique(pref_oris))
    counts = np.array([np.sum(pref_oris == ori) for ori in unique_oris])
    
    ax.bar(unique_oris, counts, width=np.diff(unique_oris).min() * 0.8 if len(unique_oris) > 1 else 1.0,
           color='mediumorchid', edgecolor='black', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Orientation (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
    
    title = f'Preferred Orientation Distribution\n({len(pref_oris)} cells)'
    if mouse_name:
        title = f'{mouse_name}\n' + title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(unique_oris)
    ax.set_xticklabels([f'{int(ori)}' for ori in unique_oris])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filename = 'combined_pref_ori_distribution.png' if mouse_name == 'Combined' else 'pref_ori_distribution.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_pref_tf(df, output_dir, mouse_name=None):
    """Plot preferred temporal frequency distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    pref_tfs = df['pref_tf'].dropna().values
    
    if len(pref_tfs) == 0:
        plt.close()
        return
    
    unique_tfs = np.sort(np.unique(pref_tfs))
    counts = np.array([np.sum(pref_tfs == tf) for tf in unique_tfs])
    
    x_positions = np.arange(len(unique_tfs))
    ax.bar(x_positions, counts, width=0.8,
           color='darkorange', edgecolor='black', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Temporal Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
    
    title = f'Preferred Temporal Frequency Distribution\n({len(pref_tfs)} cells)'
    if mouse_name:
        title = f'{mouse_name}\n' + title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{tf:.1f}' for tf in unique_tfs])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filename = 'combined_pref_tf_distribution.png' if mouse_name == 'Combined' else 'pref_tf_distribution.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_pref_sf(df, output_dir, mouse_name=None):
    """Plot preferred spatial frequency distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    pref_sfs = df['pref_sf'].dropna().values
    
    if len(pref_sfs) == 0:
        plt.close()
        return
    
    unique_sfs = np.sort(np.unique(pref_sfs))
    counts = np.array([np.sum(pref_sfs == sf) for sf in unique_sfs])
    
    x_positions = np.arange(len(unique_sfs))
    ax.bar(x_positions, counts, width=0.8,
           color='orangered', edgecolor='black', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Spatial Frequency (cpd)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
    
    title = f'Preferred Spatial Frequency Distribution\n({len(pref_sfs)} cells)'
    if mouse_name:
        title = f'{mouse_name}\n' + title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{sf:.2f}' for sf in unique_sfs])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filename = 'combined_pref_sf_distribution.png' if mouse_name == 'Combined' else 'pref_sf_distribution.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_figures(df, output_dir):
    """
    Create summary scatter plots of preferred metrics vs RF position.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    output_dir : Path
        Output directory
    """
    metrics = ['pref_ori', 'pref_tf', 'pref_sf', 'osi', 'dsi']
    metric_labels = {
        'pref_ori': 'Preferred Orientation (deg)',
        'pref_tf': 'Preferred Temporal Frequency (Hz)',
        'pref_sf': 'Preferred Spatial Frequency (cpd)',
        'osi': 'Orientation Selectivity Index',
        'dsi': 'Direction Selectivity Index'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Scatter plot colored by metric value
        scatter = ax.scatter(
            df['rf_x_center'],
            df['rf_y_center'],
            c=df[metric],
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        
        ax.set_xlabel('RF X Center', fontsize=12)
        ax.set_ylabel('RF Y Center', fontsize=12)
        ax.set_title(f'RF Position vs {metric_labels[metric]}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label=metric_labels[metric])
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    # Add overall title
    fig.suptitle('Preferred Metrics vs Receptive Field Position', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_rf_position_vs_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_rf_examples(df, output_dir, n_examples=20):
    """
    Plot example receptive fields (requires saved RF data).
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    output_dir : Path
        Output directory
    n_examples : int
        Number of examples to plot
    """
    # This is a placeholder - actual implementation would require loading RF data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'RF examples would be plotted here\n(requires raw RF data)',
            ha='center', va='center', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(output_dir / 'rf_examples.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(df, output_dir):
    """
    Plot correlation matrix of metrics.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    output_dir : Path
        Output directory
    """
    metrics = ['pref_ori', 'pref_tf', 'pref_sf', 'osi', 'dsi', 'rf_x_center', 'rf_y_center']
    
    # Select only numeric columns that exist
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) < 2:
        return
    
    # Calculate correlation matrix
    corr_matrix = df[available_metrics].corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Correlation Matrix of Metrics', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_mouse_comparison(df, output_dir):
    """
    Create comparison plots across mice.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    output_dir : Path
        Output directory
    """
    if 'mouse_name' not in df.columns or df['mouse_name'].nunique() < 2:
        return
    
    metrics = ['osi', 'dsi', 'pref_ori']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Box plot by mouse
        df.boxplot(column=metric, by='mouse_name', ax=ax)
        ax.set_xlabel('Mouse', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} by Mouse', fontsize=14, fontweight='bold')
        ax.get_figure().suptitle('')  # Remove default title
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mouse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_report(df, output_dir, args):
    """
    Create a text summary report.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    output_dir : Path
        Output directory
    args : Namespace
        Command line arguments
    """
    report_path = output_dir / 'analysis_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PREFERRED METRICS ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Analysis Date: {output_dir.name}\n")
        f.write(f"Probe: {args.probe}\n")
        f.write(f"Filtering: ")
        if args.filtered:
            f.write(f"Gaussian fit (R² >= {args.r2_threshold})\n")
        elif args.all_units:
            f.write("All units (no SNR filter)\n")
        else:
            f.write("SNR > 1 only\n")
        f.write("\n")
        
        f.write(f"Total units analyzed: {len(df)}\n")
        f.write(f"Number of mice: {df['mouse_name'].nunique()}\n")
        f.write(f"Mice: {', '.join(sorted(df['mouse_name'].unique()))}\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("METRIC STATISTICS (COMBINED)\n")
        f.write("-" * 80 + "\n\n")
        
        metrics = ['pref_ori', 'pref_tf', 'pref_sf', 'osi', 'dsi']
        for metric in metrics:
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Mean: {df[metric].mean():.3f}\n")
            f.write(f"  Median: {df[metric].median():.3f}\n")
            f.write(f"  Std: {df[metric].std():.3f}\n")
            f.write(f"  Min: {df[metric].min():.3f}\n")
            f.write(f"  Max: {df[metric].max():.3f}\n")
            f.write("\n")
        
        # Per-mouse statistics
        f.write("-" * 80 + "\n")
        f.write("METRIC STATISTICS (PER MOUSE)\n")
        f.write("-" * 80 + "\n\n")
        
        for mouse in sorted(df['mouse_name'].unique()):
            mouse_df = df[df['mouse_name'] == mouse]
            f.write(f"{mouse} (n={len(mouse_df)}):\n")
            for metric in metrics:
                f.write(f"  {metric}: mean={mouse_df[metric].mean():.3f}, "
                       f"median={mouse_df[metric].median():.3f}\n")
            f.write("\n")
        
        if 'r_squared' in df.columns:
            f.write("-" * 80 + "\n")
            f.write("GAUSSIAN FIT QUALITY (R²)\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"  Mean: {df['r_squared'].mean():.3f}\n")
            f.write(f"  Median: {df['r_squared'].median():.3f}\n")
            f.write(f"  Min: {df['r_squared'].min():.3f}\n")
            f.write(f"  Max: {df['r_squared'].max():.3f}\n")
            f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 80 + "\n\n")
        f.write("  - results.csv: Full results table\n")
        f.write("  - combined_*.png: Plots combined across all mice\n")
        f.write("  - <mouse_name>/*.png: Individual plots for each mouse\n")
        f.write("  - summary_rf_position_vs_metrics.png: RF position scatter plots\n")
        f.write("  - correlation_matrix.png: Metric correlations\n")
        if df['mouse_name'].nunique() > 1:
            f.write("  - mouse_comparison.png: Cross-mouse comparisons\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")