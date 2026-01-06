import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. Parameters
# ============================================================

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "TSLA", "KO", "PG", "JPM"
]

START_DATE = "2018-01-01"
END_DATE = "2023-12-31"

N_COMPONENTS_PCA = 3          # dimension latente
SEGMENT_LENGTH = 21           # ~ 1 mois de trading
RANDOM_SEED = 0

np.random.seed(RANDOM_SEED)

# ============================================================
# 2. Download prices
# ============================================================

prices = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False
)["Close"]

prices = prices.dropna()

# ============================================================
# 3. Log-returns
# ============================================================

log_prices = np.log(prices)
returns = log_prices.diff().dropna()


# shape: (T, num_assets)
print("Returns shape:", returns.shape)

# ============================================================
# 4. Standardization
# ============================================================

scaler = StandardScaler()
returns_std = scaler.fit_transform(returns.values)

# ============================================================
# 5. PCA
# ============================================================

pca = PCA(n_components=N_COMPONENTS_PCA)
x = pca.fit_transform(returns_std)

print("Explained variance ratio:",
      np.round(pca.explained_variance_ratio_.sum(), 3))

# ============================================================
# 6. Build TCL segments (u_t)
# ============================================================

T = x.shape[0]
num_segment = T // SEGMENT_LENGTH

u = np.repeat(np.arange(num_segment), SEGMENT_LENGTH)

# truncate x to match u
x = x[:len(u)]

print("Number of segments:", num_segment)
print("Segment length:", SEGMENT_LENGTH)

# ============================================================
# 7. Final formatting
# ============================================================

# x : (T, d)
# u : (T,)
assert x.shape[0] == u.shape[0]

# Save to disk
np.save("x_finance.npy", x)
np.save("u_finance.npy", u)

print("Saved:")
print("  x_finance.npy", x.shape)
print("  u_finance.npy", u.shape)
