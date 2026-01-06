"""
Analyze IIA-GCL Innovations from Financial Data

This script analyzes the innovations extracted by IIA-GCL:
1. Time series visualization
2. Correlation analysis (independence check)
3. Nonstationarity analysis (variance over time)
4. Correlation with market returns
5. Rolling statistics
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_innovations_timeseries(h: np.ndarray, outdir: str, n_plot: int = 5) -> None:
    """Plot time series of innovations."""
    n_plot = min(n_plot, h.shape[1])
    safe_makedirs(outdir)
    
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 2.5 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    
    for i in range(n_plot):
        axes[i].plot(h[:, i], linewidth=0.7)
        axes[i].set_ylabel(f"Innov. {i}")
        axes[i].axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    axes[-1].set_xlabel("Time index")
    plt.suptitle("Extracted Innovations (IIA-GCL)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "innovations_timeseries.png"), dpi=200)
    plt.savefig(os.path.join(outdir, "innovations_timeseries.pdf"))
    plt.close()
    print(f"  Saved: innovations_timeseries.png")


def plot_correlation_matrix(h: np.ndarray, outdir: str) -> None:
    """Plot correlation matrix of innovations."""
    safe_makedirs(outdir)
    
    corr = np.corrcoef(h, rowvar=False)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add correlation values
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if i != j:
                ax.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center', fontsize=8)
    
    ax.set_xlabel("Innovation component")
    ax.set_ylabel("Innovation component")
    ax.set_title("Correlation Matrix of Innovations\n(Should be close to diagonal)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "innovations_correlation.png"), dpi=200)
    plt.savefig(os.path.join(outdir, "innovations_correlation.pdf"))
    plt.close()
    
    # Print summary
    off_diag = corr[np.triu_indices_from(corr, k=1)]
    print(f"  Off-diagonal correlations: mean={np.mean(np.abs(off_diag)):.4f}, max={np.max(np.abs(off_diag)):.4f}")
    print(f"  Saved: innovations_correlation.png")


def plot_rolling_variance(h: np.ndarray, outdir: str, window: int = 50) -> None:
    """Plot rolling variance to visualize nonstationarity."""
    safe_makedirs(outdir)
    
    n_comp = h.shape[1]
    
    fig, axes = plt.subplots(n_comp, 1, figsize=(14, 2 * n_comp), sharex=True)
    if n_comp == 1:
        axes = [axes]
    
    for i in range(n_comp):
        # Compute rolling variance
        series = pd.Series(h[:, i])
        rolling_var = series.rolling(window=window, center=True).var()
        
        axes[i].plot(rolling_var.values, linewidth=0.8)
        axes[i].set_ylabel(f"Var(Innov. {i})")
        axes[i].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Target=1')
    
    axes[-1].set_xlabel("Time index")
    plt.suptitle(f"Rolling Variance (window={window}) - Nonstationarity Check", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "innovations_rolling_variance.png"), dpi=200)
    plt.savefig(os.path.join(outdir, "innovations_rolling_variance.pdf"))
    plt.close()
    print(f"  Saved: innovations_rolling_variance.png")


def plot_histograms(h: np.ndarray, outdir: str) -> None:
    """Plot histograms of innovations."""
    safe_makedirs(outdir)
    
    n_comp = h.shape[1]
    n_cols = min(3, n_comp)
    n_rows = (n_comp + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for i in range(n_comp):
        ax = axes[i]
        ax.hist(h[:, i], bins=50, density=True, alpha=0.7, edgecolor='black')
        
        # Overlay normal distribution
        x_range = np.linspace(h[:, i].min(), h[:, i].max(), 100)
        mu, std = h[:, i].mean(), h[:, i].std()
        ax.plot(x_range, stats.norm.pdf(x_range, mu, std), 'r-', linewidth=2, label='Normal')
        
        ax.set_title(f"Innovation {i}\nμ={mu:.2f}, σ={std:.2f}")
        ax.legend(fontsize=8)
    
    # Hide empty subplots
    for i in range(n_comp, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Distribution of Innovations", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "innovations_histograms.png"), dpi=200)
    plt.savefig(os.path.join(outdir, "innovations_histograms.pdf"))
    plt.close()
    print(f"  Saved: innovations_histograms.png")


def compute_nonstationarity_stats(h: np.ndarray, n_segments: int = 10) -> pd.DataFrame:
    """
    Compute nonstationarity statistics for each innovation component.
    Divides data into segments and compares variance across segments.
    """
    T, d = h.shape
    segment_len = T // n_segments
    
    results = []
    
    for i in range(d):
        segment_vars = []
        segment_means = []
        
        for s in range(n_segments):
            start = s * segment_len
            end = (s + 1) * segment_len if s < n_segments - 1 else T
            segment = h[start:end, i]
            segment_vars.append(np.var(segment))
            segment_means.append(np.mean(segment))
        
        # Coefficient of variation of variance
        cv_var = np.std(segment_vars) / (np.mean(segment_vars) + 1e-12)
        
        # Range of variance (log scale)
        log_var_range = np.log(max(segment_vars) / (min(segment_vars) + 1e-12))
        
        # Overall statistics
        overall_mean = np.mean(h[:, i])
        overall_std = np.std(h[:, i])
        
        # Normality test (Jarque-Bera)
        jb_stat, jb_pval = stats.jarque_bera(h[:, i])
        
        results.append({
            'component': i,
            'mean': overall_mean,
            'std': overall_std,
            'cv_variance': cv_var,
            'log_var_range': log_var_range,
            'jb_statistic': jb_stat,
            'jb_pvalue': jb_pval,
            'is_normal_5pct': jb_pval > 0.05
        })
    
    return pd.DataFrame(results)


def analyze_market_correlation(h: np.ndarray, x: np.ndarray, outdir: str) -> None:
    """Analyze correlation between innovations and original data/market proxy."""
    safe_makedirs(outdir)
    
    # Use first PC as market proxy (typically captures market factor)
    market_proxy = x[1:, 0]  # Align with h (which starts from t=1 due to AR)
    market_proxy = market_proxy[:h.shape[0]]
    
    correlations = []
    for i in range(h.shape[1]):
        corr_level = np.corrcoef(h[:, i], market_proxy)[0, 1]
        corr_abs = np.corrcoef(np.abs(h[:, i]), np.abs(market_proxy))[0, 1]
        correlations.append({
            'component': i,
            'corr_with_market': corr_level,
            'corr_abs_with_abs_market': corr_abs
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(os.path.join(outdir, "market_correlation.csv"), index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(range(h.shape[1]), corr_df['corr_with_market'])
    axes[0].set_xlabel("Innovation component")
    axes[0].set_ylabel("Correlation")
    axes[0].set_title("Correlation with Market Factor (PC1)")
    axes[0].axhline(0, color='gray', linestyle='--')
    
    axes[1].bar(range(h.shape[1]), corr_df['corr_abs_with_abs_market'])
    axes[1].set_xlabel("Innovation component")
    axes[1].set_ylabel("Correlation")
    axes[1].set_title("Correlation of |Innovation| with |Market|")
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "market_correlation.png"), dpi=200)
    plt.close()
    print(f"  Saved: market_correlation.png")
    
    return corr_df


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze IIA-GCL innovations")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to model directory. If not specified, uses most recent.")
    parser.add_argument("--n_plot", type=int, default=5,
                        help="Number of innovation components to plot")
    parser.add_argument("--rolling_window", type=int, default=50,
                        help="Window size for rolling variance")
    args = parser.parse_args()
    
    # Find model directory
    if args.model_dir is None:
        import glob
        model_dirs = sorted(glob.glob("./storage/model_igcl_*"))
        if len(model_dirs) == 0:
            raise FileNotFoundError("No IIA-GCL model found. Run training first.")
        model_dir = model_dirs[-1]
    else:
        model_dir = args.model_dir
    
    print(f"Analyzing model: {model_dir}")
    
    # Load innovations
    h_path = os.path.join(model_dir, "innovations_hat.npy")
    if not os.path.exists(h_path):
        raise FileNotFoundError(f"innovations_hat.npy not found in {model_dir}. Run evaluation first.")
    
    h = np.load(h_path)
    print(f"Loaded innovations: shape = {h.shape}")
    
    # Load original data for comparison
    x = np.load("x_finance.npy")
    
    # Output directory
    outdir = os.path.join(model_dir, "figures")
    safe_makedirs(outdir)
    print(f"Output directory: {outdir}\n")
    
    # =========================================================================
    # Generate all analyses
    # =========================================================================
    
    print("Generating visualizations...")
    
    # 1. Time series
    plot_innovations_timeseries(h, outdir, n_plot=args.n_plot)
    
    # 2. Correlation matrix
    plot_correlation_matrix(h, outdir)
    
    # 3. Rolling variance
    plot_rolling_variance(h, outdir, window=args.rolling_window)
    
    # 4. Histograms
    plot_histograms(h, outdir)
    
    # 5. Nonstationarity statistics
    print("\nComputing nonstationarity statistics...")
    stats_df = compute_nonstationarity_stats(h)
    stats_df.to_csv(os.path.join(outdir, "nonstationarity_stats.csv"), index=False)
    print(stats_df.to_string(index=False))
    
    # 6. Market correlation
    print("\nAnalyzing market correlation...")
    market_corr = analyze_market_correlation(h, x, outdir)
    print(market_corr.to_string(index=False))
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Independence check
    corr_matrix = np.corrcoef(h, rowvar=False)
    off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    print(f"Independence (off-diagonal correlations):")
    print(f"  Mean |correlation|: {np.mean(np.abs(off_diag)):.4f}")
    print(f"  Max |correlation|:  {np.max(np.abs(off_diag)):.4f}")
    
    # Nonstationarity check
    print(f"\nNonstationarity (CV of segment variance):")
    print(f"  Mean CV: {stats_df['cv_variance'].mean():.4f}")
    print(f"  Max CV:  {stats_df['cv_variance'].max():.4f}")
    
    # Normality check
    n_normal = stats_df['is_normal_5pct'].sum()
    print(f"\nNormality (Jarque-Bera test at 5%):")
    print(f"  {n_normal}/{h.shape[1]} components appear Gaussian")
    
    print(f"\nAll results saved to: {outdir}")
    print("Done!")


if __name__ == "__main__":
    main()
