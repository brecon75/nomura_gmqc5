import pandas as pd
from typing import List

try:
    df = pd.read_csv('trade_data.csv')
except FileNotFoundError:
    df = pd.DataFrame()

def expected_pnl(client: str, tau: List[int]) -> dict:
    """
    Computes the expected PnL per trade at each horizon and the aggregate PnL.
    
    Parameters:
    client: Client identifier
    tau: List of horizons e.g. [5, 10, 15, 20, 25, 30]
    
    Returns:
    Dictionary with keys 'per_horizon' and 'aggregate'
    """
    if df.empty:
        return {'per_horizon': [0.0]*len(tau), 'aggregate': 0.0}
        
    client_data = df[df['Name'] == client]  
    per_horizon = []
    for t in tau:
        # Eq 5: PnL(tau) = side * V * (Mt - Tp)
        pnl_t = client_data['Side'] * client_data['Volume'] * (client_data[f'M{t}'] - client_data['Trade Price'])
        per_horizon.append(float(pnl_t.mean()))
        
    # Eq 6: Aggregate PnL = sum( side * V * (Mt - Tp) / 6 ) for all t in {5, 10, 15, 20, 25, 30}
    agg_pnl = 0
    all_taus = [5, 10, 15, 20, 25, 30]
    
    # Vectorized computation across columns instead of evaluating row-wise logic iteratively
    mid_diffs = client_data[[f'M{t}' for t in all_taus]].sub(client_data['Trade Price'], axis=0)
    pnl_matrix = mid_diffs.mul(client_data['Side'] * client_data['Volume'], axis=0)
    agg_pnl = pnl_matrix.sum(axis=1) / 6.0
        
    return {
        'per_horizon': per_horizon,
        'aggregate': float(agg_pnl.mean())
    }

def classify_client(client: str) -> str:
    # Classifies as costly or profitable
    res = expected_pnl(client, [5, 10, 15, 20, 25, 30])
    return 'profitable' if res['aggregate'] >= 0 else 'costly'

def min_half_spread(client: str) -> float:
    #Calculates the minimum half-spread (delta) such that the expected aggregate PnL >= 0  
    client_data = df[df['Name'] == client]
        
    # Calculate M_avg = sum(M_t) / 6
    all_taus = [5, 10, 15, 20, 25, 30]
    m_avg = client_data[[f'M{t}' for t in all_taus]].mean(axis=1)
    
    m0 = client_data['M0']
    side = client_data['Side']
    v = client_data['Volume']
    
    # Derivation for new trade price under quoting rule: Tp_new = M0 - side * delta
    # New Agg PnL = side * V * (M_avg - Tp_new) = side * V * (M_avg - M0) + V * delta
    # To make Expected Agg PnL >= 0: E[V * delta + side * V * (M_avg - M0)] >= 0
    # delta >= E[-side * V * (M_avg - M0)] / E[V]
    
    expected_volume = v.mean()
        
    expected_loss_component = (-side * v * (m_avg - m0)).mean()
    
    delta = expected_loss_component / expected_volume
    
    # Minimum half-spread conceptually should be floored at 0
    return max(0.0, float(delta))


#output
if __name__ == "__main__":
    if not df.empty:
        clients = sorted(df['Name'].unique())
        horizons = [5, 10, 15, 20, 25, 30]
        
        results = []
        for c in clients:
            pnl_metrics = expected_pnl(c, horizons)
            delta_star = min_half_spread(c)
            
            row_data = {'client': c}
            for t, pnl_val in zip(horizons, pnl_metrics['per_horizon']):
                row_data[f'tau={t}'] = pnl_val
                
            row_data['agg_pnl'] = pnl_metrics['aggregate']
            row_data['delta^*'] = delta_star
            
            results.append(row_data)
            
        results_df = pd.DataFrame(results)
        
        cols = ['client', 'tau=5', 'tau=10', 'tau=15', 'tau=20', 'tau=25', 'tau=30', 'agg_pnl', 'delta^*']
        results_df = results_df[cols]
        
        results_df.to_csv('task2_results.csv', index=False)
        print("Successfully generated task2_results.csv")
    else:
        print("Data file 'trade_data.csv' not found. Cannot generate results.")