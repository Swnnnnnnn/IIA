"""
analyze_innovations.py

Loads innovations extracted by iia_evaluation_real.py and produces figures and
simple quantitative summaries for a report.

Outputs are saved in <model_dir>/figures by default:
1) Time series plots of a few innovations
2) Segment-wise variance heatmap
3) Innovation correlation heatmap
4) Optional correlation with market returns and market volatility proxy
5) Simple nonstationarity scores saved to CSV
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def zscore(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = np.nanmean(a, axis=0, keepdims=True)
    sd = np.nanstd(a, axis=0, keepdims=True)
    return (a - mu) / (sd + eps)


def corrcoef_cols(a: np.ndarray) -> np.ndarray:
    a0 = a - np.mean(a, axis=0, keepdims=True)
    denom = np.sqrt(np.sum(a0 * a0, axis=0, keepdims=True)) + 1e-12
    a0 = a0 / denom
    return (a0.T @ a0) / (a0.shape[0] - 1)


def segment_stats(h: np.ndarray, seg: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two dataframes:
    mean_df: index=segment, columns=comp_i
    var_df:  index=segment, columns=comp_i
    """
    comps = [f"comp_{i}" for i in range(h.shape[1])]
    df = pd.DataFrame(h, columns=comps)
    df["segment"] = seg

    mean_df = df.groupby("segment")[comps].mean()
    var_df = df.groupby("segment")[comps].var(ddof=1)

    return mean_df, var_df


def nonstationarity_scores(var_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple scores derived from segment variances per component.
    cv_var: coefficient of variation of segment variances
    log_range: log(max var / min var)
    """
    eps = 1e-12
    v = var_df.values  # shape (S, d)

    v_mean = np.nanmean(v, axis=0)
    v_std = np.nanstd(v, axis=0)
    cv_var = v_std / (v_mean + eps)

    v_min = np.nanmin(v, axis=0)
    v_max = np.nanmax(v, axis=0)
    log_range = np.log((v_max + eps) / (v_min + eps))

    out = pd.DataFrame(
        {
            "component": var_df.columns,
            "cv_var": cv_var,
            "log_var_range": log_range,
            "mean_var": v_mean,
            "min_var": v_min,
            "max_var": v_max,
        }
    )
    return out.sort_values("cv_var", ascending=False).reset_index(drop=True)


def plot_innovations(h: np.ndarray, outdir: str, n_plot: int = 5) -> None:
    n_plot = int(min(n_plot, h.shape[1]))
    safe_makedirs(outdir)

    for i in range(n_plot):
        plt.figure(figsize=(11, 3.2))
        plt.plot(h[:, i])
        plt.title(f"Innovation comp_{i}")
        plt.xlabel("time index")
        plt.ylabel("value")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"innovation_comp_{i}.png"), dpi=200)
        plt.savefig(os.path.join(outdir, f"innovation_comp_{i}.pdf"))
        plt.close()


def plot_corr_heatmap(h: np.ndarray, outdir: str) -> None:
    safe_makedirs(outdir)
    c = np.corrcoef(h, rowvar=False)

    plt.figure(figsize=(7.5, 6.5))
    im = plt.imshow(c, aspect="auto")
    plt.colorbar(im)
    plt.title("Correlation matrix of innovations")
    plt.xlabel("component")
    plt.ylabel("component")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "innovations_correlation_heatmap.png"), dpi=200)
    plt.savefig(os.path.join(outdir, "innovations_correlation_heatmap.pdf"))
    plt.close()


def plot_segment_variance_heatmap(var_df: pd.DataFrame, outdir: str, log_scale: bool = True) -> None:
    safe_makedirs(outdir)
    v = var_df.values
    if log_scale:
        v = np.log(v + 1e-12)

    plt.figure(figsize=(10.5, 5.8))
    im = plt.imshow(v, aspect="auto")
    plt.colorbar(im)
    plt.title("Segment-wise variance of innovations" + (" (log)" if log_scale else ""))
    plt.xlabel("component")
    plt.ylabel("segment")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "segment_variance_heatmap.png"), dpi=200)
    plt.savefig(os.path.join(outdir, "segment_variance_heatmap.pdf"))
    plt.close()


def fetch_market_returns(start: str | None = None, end: str | None = None, ticker: str = "^GSPC") -> pd.Series | None:
    """
    Tries to fetch market close prices with yfinance and returns log returns.
    If yfinance is missing or download fails, returns None.
    """
    try:
        import yfinance as yf
    except Exception:
        return None

    try:
        px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        px = px.dropna()
        r = np.log(px).diff().dropna()
        r.name = f"{ticker}_logret"
        return r
    except Exception:
        return None


def corr_with_market(h: np.ndarray, market_ret: np.ndarray) -> pd.DataFrame:
    """
    Correlation of each innovation with market returns and abs(market returns).
    Assumes aligned lengths.
    """
    m = market_ret.reshape(-1, 1)
    m_abs = np.abs(market_ret).reshape(-1, 1)

    hz = zscore(h)
    mz = zscore(m)
    maz = zscore(m_abs)

    corr_m = (hz * mz).mean(axis=0)
    corr_abs = (hz * maz).mean(axis=0)

    return pd.DataFrame(
        {
            "component": [f"comp_{i}" for i in range(h.shape[1])],
            "corr_market_ret": corr_m,
            "corr_abs_market_ret": corr_abs,
        }
    ).sort_values("corr_market_ret", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="Path to storage/model_YYYYMMDD_HHMMSS folder")
    ap.add_argument("--n_plot", type=int, default=5, help="Number of innovation components to plot")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory for figures and tables")
    ap.add_argument("--market_ticker", type=str, default="^GSPC", help="Ticker for market benchmark, ex ^GSPC or SPY")
    ap.add_argument("--fetch_market", action="store_true", help="If set, try to fetch market returns with yfinance")
    args = ap.parse_args()

    model_dir = args.model_dir
    if args.out_dir is None:
        out_dir = os.path.join(model_dir, "figures")
    else:
        out_dir = args.out_dir
    safe_makedirs(out_dir)

    h_path = os.path.join(model_dir, "innovations_hat.npy")
    seg_path = os.path.join(model_dir, "segments.npy")
    if not (os.path.exists(h_path) and os.path.exists(seg_path)):
        raise FileNotFoundError(
            "Missing innovations_hat.npy or segments.npy in model_dir. "
            "Run iia_evaluation_real.py first."
        )

    h = np.load(h_path)
    seg = np.load(seg_path)

    if h.ndim == 1:
        h = h.reshape(-1, 1)
    if seg.ndim != 1:
        seg = seg.reshape(-1)

    n = min(h.shape[0], seg.shape[0])
    h = h[:n]
    seg = seg[:n]

    print("Loaded")
    print("  h shape:", h.shape)
    print("  segments shape:", seg.shape)
    print("  num segments:", len(np.unique(seg)))

    mean_df, var_df = segment_stats(h, seg)
    scores_df = nonstationarity_scores(var_df)

    mean_df.to_csv(os.path.join(out_dir, "segment_means.csv"))
    var_df.to_csv(os.path.join(out_dir, "segment_variances.csv"))
    scores_df.to_csv(os.path.join(out_dir, "nonstationarity_scores.csv"), index=False)

    plot_innovations(h, out_dir, n_plot=args.n_plot)
    plot_corr_heatmap(h, out_dir)
    plot_segment_variance_heatmap(var_df, out_dir, log_scale=True)

    if args.fetch_market:
        market = fetch_market_returns(ticker=args.market_ticker)
        if market is None:
            print("Could not fetch market data. Install yfinance or check connection.")
        else:
            m = market.values
            L = min(len(m), h.shape[0])
            h_m = h[-L:, :]
            m = m[-L:]

            corr_df = corr_with_market(h_m, m)
            corr_df.to_csv(os.path.join(out_dir, "corr_with_market.csv"), index=False)

            plt.figure(figsize=(8.5, 4.2))
            plt.bar(np.arange(h.shape[1]), corr_df.sort_values("component")["corr_market_ret"].values)
            plt.title(f"Correlation of innovations with {args.market_ticker} log returns")
            plt.xlabel("component")
            plt.ylabel("correlation")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "corr_with_market_returns.png"), dpi=200)
            plt.savefig(os.path.join(out_dir, "corr_with_market_returns.pdf"))
            plt.close()

    print("Saved outputs to:", out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
