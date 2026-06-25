import pandas as pd
from typing import List

# Assume trade_data.csv is in the current working directory as per standard execution
try:
    df = pd.read_csv('trade_data.csv')
except FileNotFoundError:
    df = pd.DataFrame()

def adversity_profile(client: str, tau: List[int]) -> List[float]:
    """
    Simulates the adversity profile of a client at specified horizons.
    Adversity is the percentage of trades that were adverse (PnL < 0) at horizon tau.
    """
    client_data = df[df['Name'] == client]
        
    adversity_percentages = []
    
    for t in tau:
        # PnL (at t=tau) = side * V * (Mt - Tp)
        pnl = client_data['Side'] * client_data['Volume'] * (client_data[f'M{t}'] - client_data['Trade Price'])
        
        # Calculate percentage of trades strictly < 0
        adverse_pct = (pnl < 0).mean() * 100
        adversity_percentages.append(float(adverse_pct))
        
    return adversity_percentages

if __name__ == "__main__":
    if not df.empty:
        clients = sorted(df['Name'].unique())
        horizons = [5, 10, 15, 20, 25, 30]
        
        results = []
        for c in clients:
            profile = adversity_profile(c, horizons)
            
            row_data = {'client': c}
            for t, adv_val in zip(horizons, profile):
                row_data[f'tau={t}'] = adv_val
            results.append(row_data)
            
        # Create DataFrame and save exactly as requested
        results_df = pd.DataFrame(results)
        
        # Ensure column ordering matches submission requirements
        cols = ['client', 'tau=5', 'tau=10', 'tau=15', 'tau=20', 'tau=25', 'tau=30']
        results_df = results_df[cols]
        
        results_df.to_csv('task1_results.csv', index=False)
        print("Successfully generated task1_results.csv")
    else:
        print("Data file 'trade_data.csv' not found. Cannot generate results.")