import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TAUS       = [5, 10, 15, 20, 25, 30]
TAU_TO_COL = {5: 'M5', 10: 'M10', 15: 'M15', 20: 'M20', 25: 'M25', 30: 'M30'}

# 60/20/20 split
VAL_START_FRAC  = 0.60
TEST_START_FRAC = 0.80

# threshold grid
THRESHOLDS = np.linspace(0, 1, 1001)


def optimal_threshold(df_val: pd.DataFrame, df_test: pd.DataFrame,
                      tau: int, client_specific: bool = True) -> dict:
    """
    Calculates the optimal externalization threshold to maximise PnL.

    Parameters:
        df_val          : Validation set DataFrame.
        df_test         : Test set DataFrame.
        tau             : Horizon in seconds (5, 10, 15, 20, 25, 30).
        client_specific : If True, optimises a separate theta per client.
                          If False, optimises a single global theta.

    Expected DataFrame columns:
        'client'    : str   — client identifier
        'pred_prob' : float — adversity probability from Task 3 model
        'actual_pnl': float — LP PnL per trade at horizon tau (Eq. 5)

    Returns:
        dict with keys:
            'theta'          : float (global) or Dict[str, float] (per-client)
            'validation_pnl' : float — PnL at optimal theta on validation set
            'test_pnl'       : float — PnL at optimal theta on test set
    """
    if not client_specific:
        # Global best theta
        best_theta   = 1.0
        max_val_pnl  = -np.inf

        for theta in THRESHOLDS:
            val_pnl = df_val.loc[df_val['pred_prob'] <= theta, 'actual_pnl'].sum()
            if val_pnl > max_val_pnl:
                max_val_pnl = val_pnl
                best_theta  = theta

        test_pnl = df_test.loc[df_test['pred_prob'] <= best_theta, 'actual_pnl'].sum()

        return {
            'theta'         : float(best_theta),
            'validation_pnl': float(max_val_pnl),
            'test_pnl'      : float(test_pnl),
        }

    else:
        # Client-specific best theta
        best_thetas   = {}
        total_val_pnl = 0.0
        total_test_pnl = 0.0

        for client in sorted(df_val['client'].unique()):
            c_val  = df_val[df_val['client'] == client]
            c_test = df_test[df_test['client'] == client]

            c_best_theta  = 1.0
            c_max_val_pnl = -np.inf

            for theta in THRESHOLDS:
                val_pnl = c_val.loc[c_val['pred_prob'] <= theta, 'actual_pnl'].sum()
                if val_pnl > c_max_val_pnl:
                    c_max_val_pnl = val_pnl
                    c_best_theta  = theta

            best_thetas[client] = float(c_best_theta)
            total_val_pnl  += c_max_val_pnl
            total_test_pnl += c_test.loc[
                c_test['pred_prob'] <= c_best_theta, 'actual_pnl'
            ].sum()

        return {
            'theta'         : best_thetas,
            'validation_pnl': float(total_val_pnl),
            'test_pnl'      : float(total_test_pnl),
        }


def plot_pnl_vs_theta(df_val: pd.DataFrame, tau: int) -> None:
    """
    Plots PnL_validation(theta) for theta in [0, 1] per client,
    overlays the global optimization curve, and saves the figure
    to 'pnl_vs_theta_tau_{tau}.png'.
    """
    plt.figure(figsize=(13, 8))
    clients = sorted(df_val['client'].unique())
    colors  = plt.cm.tab10(np.linspace(0, 1, len(clients)))

    # Plot individual client curves
    for idx, client in enumerate(clients):
        c_val = df_val[df_val['client'] == client]

        # Vectorised PnL curve using cumulative sort trick for speed
        c_sorted = c_val.sort_values('pred_prob')
        cum_pnl  = c_sorted['actual_pnl'].cumsum().values
        thetas   = c_sorted['pred_prob'].values

        pnl_curve = np.interp(THRESHOLDS, thetas, cum_pnl, left=0.0)

        best_idx      = int(np.argmax(pnl_curve))
        optimal_theta = float(THRESHOLDS[best_idx])
        max_pnl       = float(pnl_curve[best_idx])

        plt.plot(THRESHOLDS, pnl_curve, linewidth=1.8, color=colors[idx], alpha=0.7,
                 label=f'Client {client} ($\\theta^*$={optimal_theta:.2f})')
        plt.plot(optimal_theta, max_pnl, marker='o', color=colors[idx], markersize=6)

    # plot the Global Optimization Baseline
    global_sorted = df_val.sort_values('pred_prob')
    global_cum_pnl = global_sorted['actual_pnl'].cumsum().values
    global_thetas  = global_sorted['pred_prob'].values

    global_pnl_curve = np.interp(THRESHOLDS, global_thetas, global_cum_pnl, left=0.0)
    
    global_best_idx = int(np.argmax(global_pnl_curve))
    global_optimal_theta = float(THRESHOLDS[global_best_idx])
    global_max_pnl = float(global_pnl_curve[global_best_idx])

    # Overlay global curve as a thick dashed line
    plt.plot(THRESHOLDS, global_pnl_curve, linewidth=3, color='black', linestyle='--',
             label=f'GLOBAL AGGREGATE ($\\theta^*$={global_optimal_theta:.2f})')
    plt.plot(global_optimal_theta, global_max_pnl, marker='X', color='black', markersize=10, 
             label=f'Global Max PnL (${global_max_pnl:,.2f})')

    plt.title(f'Validation PnL vs. Externalization Threshold $\\theta$ (Horizon: {tau}s)')
    plt.xlabel('Externalization Threshold $\\theta$')
    plt.ylabel('Total Validation PnL ($)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'pnl_vs_theta_tau_{tau}.png', dpi=300)
    plt.close()
    print(f"Saved 'pnl_vs_theta_tau_{tau}.png' with Global line overlay.")


# Output

if __name__ == '__main__':
    import t3 as t3_module

    DATA_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'trade_data.csv')
    )

    print(f"Loading data from {DATA_PATH}...")
    df = t3_module.load_data(DATA_PATH)
    t3_module.train_models(df)

    X_features   = t3_module.build_features(df).values
    df['client'] = df['Name']

    # chronological boundaries — matches t3.py 60/20/20
    n             = len(df)
    val_start     = int(VAL_START_FRAC  * n)   # 60%
    test_start    = int(TEST_START_FRAC * n)   # 80%

    #csv initialisation
    csv_rows = []

    for tau in TAUS:
        print(f"\n{'='*55}")
        print(f"PROCESSING TAU = {tau}")
        print(f"{'='*55}")

        # Predict adversity probabilities for this horizon
        df['pred_prob']  = t3_module.predict_adversity(features=X_features, tau=tau)

        # LP PnL per trade at horizon tau (Eq. 5):
        # PnL = Side * Volume * (M_tau - Trade Price)
        df['actual_pnl'] = (df['Side'] * df['Volume'] *
                            (df[TAU_TO_COL[tau]] - df['Trade Price']))

        df_val  = df.iloc[val_start:test_start].copy()
        df_test = df.iloc[test_start:].copy()

        # global theta optimisation 
        global_res = optimal_threshold(df_val, df_test, tau=tau, client_specific=False)
        print(f"\nGlobal Optimisation")
        print(f"  Optimal Theta    : {global_res['theta']:.4f}")
        print(f"  Validation PnL   : ${global_res['validation_pnl']:>12,.2f}")
        print(f"  Test PnL         : ${global_res['test_pnl']:>12,.2f}")

        # client specific theta optimisation
        client_res = optimal_threshold(df_val, df_test, tau=tau, client_specific=True)
        print(f"\nClient-Specific Optimisation")
        for client, theta in client_res['theta'].items():
            c_test_pnl = df_test.loc[
                (df_test['client'] == client) &
                (df_test['pred_prob'] <= theta), 'actual_pnl'
            ].sum()
            print(f"  Client {client}: theta*={theta:.4f}  test PnL=${c_test_pnl:>10,.2f}")
            csv_rows.append({
                'client'   : client,
                'tau'      : tau,
                'theta_star': round(theta, 4),
                'final_pnl': round(c_test_pnl, 2),
            })
        print(f"  Total Validation PnL : ${client_res['validation_pnl']:>12,.2f}")
        print(f"  Total Test PnL       : ${client_res['test_pnl']:>12,.2f}")


        print(f"\nGenerating plot for tau = {tau}...")
        plot_pnl_vs_theta(df_val, tau=tau)

    # csv output
    results_df = pd.DataFrame(csv_rows)[['client', 'tau', 'theta_star', 'final_pnl']]
    results_df = results_df.sort_values(['client', 'tau']).reset_index(drop=True)
    output_csv = 'task4_results.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\n[+] Saved '{output_csv}'")
    print(results_df.to_string(index=False))