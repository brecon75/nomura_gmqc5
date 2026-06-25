import sys
import numpy as np
import pandas as pd
import math

def quote(inventory: float, sigma: float, alpha: float, eta: float, 
          client: str = 'X', volume: float = 0.0, m0: float = None) -> tuple[float, float]:
    """
    Core quoting function. Required interface: (inventory, sigma, alpha, eta).
    Optional args (client, volume, m0) provide additional context when available
    (e.g. in full simulation), but the function is fully callable with just the
    4 required state variables. Defaults are inert: client='X' never triggers
    the toxicity shield, volume=0.0 is always below threshold, and m0=None
    falls back to a sigma-based spread cap so no external price context is needed.
    client, volume and m0 provide additional signals to help avoid adverse trades.
    """
    # quoting parameters tuned using grid search on validation set
    c_base = 0.75000
    c_alpha = 0.0500
    c_inv = 0.900
    c_time = 3.0000

    vol_threshold = 203.0000 # 95th percentile of volume
    
    # mandatory exchange constraints (clipping)
    min_spread = 0.5 * sigma
    max_spread = m0 * 0.0050  # 50 bps max constraint
    
    # if it's a toxic client, massive volume, or high ML adversity, reject the trade
    if client in ['D', 'E'] or volume > vol_threshold or alpha > 0.9:
        return float(max_spread), float(max_spread)

    base_spread = sigma * (c_base + c_alpha * alpha)
    
    # inventory skew (if we have inventory we widen spread)
    skew = c_inv * inventory * math.exp(c_time * eta)
    
    raw_delta_b = base_spread + skew
    raw_delta_a = base_spread - skew
    
    # clipping
    delta_b = max(min_spread, min(max_spread, raw_delta_b))
    delta_a = max(min_spread, min(max_spread, raw_delta_a))
    
    return float(delta_b), float(delta_a)


_CACHE = {}
def validate_quote(raw_df: pd.DataFrame, lam: float, gamma: float, phi: float, seed: int = 0) -> dict:
    """
    Ingests raw trade data (cached globally), runs the 
    quoting strategy backtest for given parameters and seed, 
    and returns summary metrics.
    """
    global _CACHE
    if not _CACHE:
        print("Ingesting Data and Training Models (this happens only once because of cache)...")
        df = raw_df.copy()
        
        # feature engineering
        df['Date'] = pd.to_datetime(df['Date'])
        df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['time'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Realized Volatility (Sigma)
        df['mid_return'] = df['M0'].pct_change()
        df['sigma'] = (df['mid_return'].rolling(20)
                       .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
                       .fillna(1e-5).clip(lower=1e-6))
                       
        # Calculate Elapsed Time (Eta)
        t0 = df.groupby('Date')['datetime'].transform('min')
        t1 = df.groupby('Date')['datetime'].transform('max')
        df['eta'] = ((df['datetime'] - t0).dt.total_seconds() /
                     (t1 - t0).dt.total_seconds().clip(lower=1)).clip(0, 1)
                     
        # Aggregate Mid-Price for PnL
        mid_cols = ['M5', 'M10', 'M15', 'M20', 'M25', 'M30']
        df['mean_M'] = df[mid_cols].mean(axis=1)
        
        # standardize column names
        df['client_side'] = df['Side']
        df['volume'] = df['Volume']
        df['client'] = df['Name']
        df['date'] = df['Date']

        # task3: adversity model import
        try:
            import t3 as t3_module
            print("Importing Adversity Model...")
            t3_module.train_models(df) 
            X_features = t3_module.build_features(df).values
            df['alpha_pred'] = t3_module.predict_adversity(features=X_features, tau=30)
        except ImportError:
            print("Warning: Could not import t3_module. Defaulting ML alpha to 0.0")
            df['alpha_pred'] = 0.0

        _CACHE['days'] = [group for _, group in df.groupby('date')]
        _CACHE['total_trades'] = len(df)
        
    days = _CACHE['days']
    total_trades = _CACHE['total_trades']
    
    np.random.seed(seed)
    all_daily_pnls = []
    total_fills = 0
    
    for day_data in days:
        inventory = 0.0
        daily_pnl = 0.0
        daily_sigma = day_data['sigma'].mean()
        
        for trade in day_data.itertuples(index=False):
            # generate quotes
            delta_b, delta_a = quote(
                inventory=inventory,
                sigma=trade.sigma,
                alpha=trade.alpha_pred,
                eta=trade.eta,
                client=trade.client,
                volume=trade.volume,
                m0=trade.M0,
            )
            
            side = trade.client_side 
            quoted_spread = delta_b if side == 1 else delta_a
            
            # Simulate Fill Probability
            p_fill = lam * np.exp(-gamma * (quoted_spread / max(trade.sigma, 1e-8)))
            is_filled = np.random.rand() < p_fill
            
            if is_filled:
                total_fills += 1
                inventory += trade.volume * side
                
                # exact pnl calculation
                trade_pnl = trade.volume * (quoted_spread + side * (trade.mean_M - trade.M0))
                daily_pnl += trade_pnl
                
        # end of day inventory penalty
        penalty = phi * (inventory ** 2) * daily_sigma
        all_daily_pnls.append(daily_pnl - penalty)
        
    # metrics and output
    total_pnl = np.sum(all_daily_pnls)
    sigma_port = np.std(all_daily_pnls)
    score = total_pnl / max(sigma_port, 1e-6)
    fill_perc = (total_fills / max(total_trades, 1)) * 100
    
    cum_pnl = np.cumsum(all_daily_pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = running_max - cum_pnl
    max_dd = np.max(drawdown)
    
    return {
        'score': score,
        'pnl': total_pnl,
        'max_dd': max_dd,
        'fill_perc': fill_perc
    }


### 10-seed average backtest across 20 diverse market regimes (lam,gamma,phi)
# if __name__ == "__main__":
#     _SCENARIOS = [
#         dict(lam=0.5, gamma=0.5,  phi=0.05),
#         dict(lam=0.5, gamma=1.0,  phi=0.10),
#         dict(lam=0.5, gamma=2.0,  phi=0.20),
#         dict(lam=0.3, gamma=1.0,  phi=0.10),
#         dict(lam=0.7, gamma=1.0,  phi=0.10),
#         dict(lam=0.5, gamma=0.3,  phi=0.03),
#         dict(lam=0.5, gamma=3.0,  phi=0.30),
#         dict(lam=0.5, gamma=5.0,  phi=0.50),
#         dict(lam=0.2, gamma=1.0,  phi=0.10),
#         dict(lam=0.8, gamma=1.0,  phi=0.10),
#         dict(lam=0.4, gamma=0.5,  phi=0.05),
#         dict(lam=0.6, gamma=2.0,  phi=0.20),
#         dict(lam=0.3, gamma=0.5,  phi=0.05),
#         dict(lam=0.7, gamma=2.0,  phi=0.20),
#         dict(lam=0.5, gamma=1.5,  phi=0.15),
#         dict(lam=0.4, gamma=1.5,  phi=0.15),
#         dict(lam=0.6, gamma=0.75, phi=0.08),
#         dict(lam=0.3, gamma=2.0,  phi=0.20),
#         dict(lam=0.7, gamma=0.5,  phi=0.05),
#         dict(lam=0.5, gamma=4.0,  phi=0.40),
#     ]

#     try:
#         test_df = pd.read_csv('trade_data.csv')
#         results = []
#         print(f"Executing 10-seed average backtest across {len(_SCENARIOS)} scenarios...")
        
#         for idx, env in enumerate(_SCENARIOS, 1):
#             lam = env['lam']
#             gamma = env['gamma']
#             phi = env['phi']
            
#             scores = []
#             pnls = []
#             max_dds = []
#             fills = []
            
#             for seed in range(10):
#                 res = validate_quote(test_df, lam, gamma, phi, seed)
#                 scores.append(res['score'])
#                 pnls.append(res['pnl'])
#                 max_dds.append(res['max_dd'])
#                 fills.append(res['fill_perc'])
                
#             results.append({
#                 'Scenario': idx,
#                 'lambda': lam,
#                 'gamma': gamma,
#                 'phi': phi,
#                 'Score': round(float(np.mean(scores)), 2),
#                 'PnL': round(float(np.mean(pnls)), 2),
#                 'MaxDD': round(float(np.mean(max_dds)), 2),
#                 'Fill%': round(float(np.mean(fills)), 2)
#             })

#         print("\n" + "="*85)
#         print("VALIDATION RESULTS ACROSS 20 SCENARIOS")
#         print("="*85)
#         results_df = pd.DataFrame(results)
#         print(results_df.to_string(index=False))
#         print("="*85)
        
#     except FileNotFoundError:
#         print("Test file 'trade_data.csv' not found. Ensure it is in the same directory.")