import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch.unitroot import VarianceRatio
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import jarque_bera

df = yf.download("^FCHI", start="1995-01-01", progress=False)  # auto_adjust=True by default
prices = df["Close"].dropna()    # already adjusted

returns = np.log(prices).diff().dropna()
returns.name = "LogRet"

out = pd.concat([prices, returns], axis=1)
out.columns = ["AdjClose", "LogRet"]
out.to_csv("cac40_daily.csv", index=True)

# Split into two subsamples for the additional analysis
mid_date = "2008-12-31"
returns_pre = returns[returns.index <= mid_date]
returns_post = returns[returns.index > mid_date]

print(df.columns)

if isinstance(returns, pd.DataFrame):
    returns = returns.iloc[:, 0]  # convert to series if it's a Dataframe

desc = returns.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
desc = desc.to_frame(name="LogReturn")

desc.loc["skew"] = returns.skew()
desc.loc["kurtosis"] = returns.kurtosis()
jb_stat, jb_p = jarque_bera(returns)
desc.loc["Jarque–Bera stat"] = jb_stat
desc.loc["p-value (JB)"] = jb_p

print("\n Descriptive Statistics")
print(desc)

# Ljung–Box test for autocorrelation
lb = acorr_ljungbox(returns, lags=[10, 20], return_df=True)
print("\n Ljung–Box test for autocorrelation")
print(lb)

# Variance ratio test (Lo–MacKinlay) 
rows = []
for k in [2, 4, 8, 16]:
    vr = VarianceRatio(returns, lags=k)
    rows.append({"k": k, "VR": vr.vr, "z-stat": vr.stat, "p-value": vr.pvalue})
vr_table = pd.DataFrame(rows)
print("\n Variance Ratio test")
print(vr_table)

from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# --- Autocorrelation Function (ACF) ---
plt.figure(figsize=(10, 5))
plot_acf(returns, lags=30, color="blue", zero=False)
plt.title("CAC 40: Autocorrelation of Daily Log Returns", fontsize=12, weight="bold")
plt.xlabel("Lag (Days)", fontsize=10)
plt.ylabel("Autocorrelation", fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("acf_returns.png", dpi=200)
plt.show()

rolling_vol = returns.rolling(30).std() * np.sqrt(252)

plt.figure(figsize=(8, 5))
plt.plot(rolling_vol, color="blue", linewidth=1.2)
plt.title("CAC40 Rolling Volatility (30-Day Annualized)", fontsize=12, weight="bold")
plt.ylabel("Annualized Volatility", fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("rolling_vol.png", dpi=200)
plt.show()

from statsmodels.sandbox.stats.runs import runstest_1samp

z_stat, p_value = runstest_1samp(returns)
print(f"Runs test: z = {z_stat:.3f}, p = {p_value:.4f}")

#Additional analysis before and after 2008
from statsmodels.graphics.tsaplots import plot_acf

def test_efficiency(subset, label):
    print(f"\n {label}")
    
    # Descriptive stats
    print(subset.describe())
    jb_stat, jb_p = jarque_bera(subset)
    print(f"Jarque–Bera p = {jb_p:.4f}")
    
    # Ljung–Box
    lb = acorr_ljungbox(subset, lags=[10,20], return_df=True)
    print("\nLjung–Box:\n", lb)
    
    # Variance Ratio
    rows=[]
    for k in [2,4,8,16]:
        vr = VarianceRatio(subset, lags=k)
        rows.append({"k":k, "VR":vr.vr, "z":vr.stat, "p":vr.pvalue})
    print("\nVariance Ratio:\n", pd.DataFrame(rows))
    
    # Runs test (optional)
    from statsmodels.sandbox.stats.runs import runstest_1samp
    z, p = runstest_1samp(subset)
    print(f"Runs test: z={z:.3f}, p={p:.4f}")
    
test_efficiency(returns_pre, "Pre-2008 (1995–2008)")
test_efficiency(returns_post, "Post-2008 (2009–2025)")




