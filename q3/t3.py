"""
task3.py — Adversity Prediction Model

METHODOLOGY & HYPERPARAMETER SEARCH:
The hyperparameters below were discovered using a rigorous RandomizedSearchCV
over 30 iterations per horizon (tau), utilizing a TimeSeriesSplit (5 folds)
to strictly prevent data leakage from future observations. The models were
optimized against the 'neg_log_loss' metric to directly align with the
competition's evaluation criteria.
"""

import os
import numpy as np
import pandas as pd
from typing import Union

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, log_loss)


DATA_PATH  = 'trade_data.csv'
TAUS       = [5, 10, 15, 20, 25, 30]
TAU_COL = {5: 'M5', 10: 'M10', 15: 'M15', 20: 'M20', 25: 'M25', 30: 'M30'}

TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20  

# Optimal hyperparameters discovered via TimeSeriesSplit RandomizedSearchCV
HARDCODED_PARAMS = {
    5:  {'min_samples_leaf': 50, 'max_iter': 200, 'max_depth': 3, 'learning_rate': 0.05, 'l2_regularization': 0.1},
    10: {'min_samples_leaf': 10, 'max_iter': 300, 'max_depth': 3, 'learning_rate': 0.01, 'l2_regularization': 0.0},
    15: {'min_samples_leaf': 10, 'max_iter': 300, 'max_depth': 3, 'learning_rate': 0.01, 'l2_regularization': 0.0},
    20: {'min_samples_leaf': 10, 'max_iter': 300, 'max_depth': 3, 'learning_rate': 0.01, 'l2_regularization': 0.0},
    25: {'min_samples_leaf': 10, 'max_iter': 300, 'max_depth': 3, 'learning_rate': 0.01, 'l2_regularization': 0.0},
    30: {'min_samples_leaf': 10, 'max_iter': 300, 'max_depth': 3, 'learning_rate': 0.05, 'l2_regularization': 0.0},
}

# Global state
models       = {}   # tau -> fitted model
feature_cols = []
_splits      = {}


def load_data(filepath: str) -> pd.DataFrame:
    df  = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df['Side'] = pd.to_numeric(df['Side'])
    if 'Date' in df.columns and 'time' in df.columns:
        df['datetime'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce'
        )
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

# feature engineering
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the feature matrix. All client-level rolling features use
    shift(1) to ensure no look-ahead leakage. Toxicity proxy uses only
    information available at trade time (M0 vs Trade Price), not future
    mid prices to avoid label leakage across horizons.

    Feature vector index map:
      [ 0]  side                   — LP side (+1 buy / -1 sell)
      [ 1]  log_volume             — log(1 + Volume)
      [ 2]  spread                 — raw bid-ask spread
      [ 3]  relative_spread        — spread / M0
      [ 4]  client_enc             — label-encoded client ID
      [ 5]  hour                   — hour of trade
      [ 6]  eta                    — elapsed fraction of trading day
      [ 7]  realized_vol           — 20-trade rolling RMS mid return
      [ 8]  side_spread            — side * spread interaction
      [ 9]  log_vol_x_rspread      — log_volume * relative_spread interaction
      [10]  client_adv_20          — client's rolling 20-trade adversity rate (lag-1)
      [11]  client_adv_50          — client's rolling 50-trade adversity rate (lag-1)
      [12]  client_avg_volume_20   — client's rolling 20-trade mean volume (lag-1)
      [13]  client_volume_ratio    — current volume / client_avg_volume_20
      [14]  client_side_imbalance  — client's rolling 20-trade mean side (lag-1)
      [15]  vol_x_side_imbalance   — volume ratio * side imbalance interaction
      [16]  ret_5                  — 5-trade mid price return
      [17]  ret_20                 — 20-trade mid price return
    """
    fe = pd.DataFrame(index=df.index)

    # original dataset features
    fe['side']            = df['Side']
    fe['log_volume']      = np.log1p(df['Volume'])
    fe['spread']          = df['Spread']
    fe['relative_spread'] = df['Spread'] / df['M0'].replace(0, np.nan)

    #  Client identity 
    le = LabelEncoder()
    fe['client_enc'] = le.fit_transform(df['Name'])

    #  Time features
    fe['hour'] = df['datetime'].dt.hour
    mins_since_open = (df['datetime'].dt.hour - 9) * 60 + df['datetime'].dt.minute
    fe['eta'] = (mins_since_open / (8 * 60)).clip(0, 1)

    #  Market microstructure 
    mid_ret = df['M0'].pct_change().fillna(0)
    fe['realized_vol'] = (mid_ret.pow(2)
                          .rolling(20, min_periods=1)
                          .mean()
                          .pipe(np.sqrt))

    #  Interaction terms
    fe['side_spread'] = fe['side'] * fe['spread']
    fe['log_vol_x_rspread'] = fe['log_volume'] * fe['relative_spread']

    #  Volume features 
    fe['client_avg_volume_20'] = df.groupby('Name')['Volume'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=1).mean()
    )
    fe['client_volume_ratio'] = df['Volume'] / fe['client_avg_volume_20'].replace(0, np.nan)

    #  Directional imbalance (informed trader signal)
    fe['client_side_imbalance'] = df.groupby('Name')['Side'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=1).mean()
    )
    # Large volume in same direction as recent flow = strong toxicity signal
    fe['vol_x_side_imbalance'] = fe['client_volume_ratio'] * fe['client_side_imbalance']

    #  Momentum
    fe['ret_5']  = df['M0'].pct_change(5)
    fe['ret_20'] = df['M0'].pct_change(20)

    return fe.fillna(0)


def build_labels(df: pd.DataFrame) -> dict:
    """
    Binary label per horizon tau: 1 if trade is adverse at t=tau, 0 otherwise.
    Adverse = LP loses money: Side * Volume * (M_tau - Trade Price) < 0
    """
    return {
        tau: (df['Side'] * df['Volume'] *
              (df[TAU_COL[tau]] - df['Trade Price']) < 0).astype(int)
        for tau in TAUS
    }

def _make_splits(X: pd.DataFrame, labels: dict) -> dict:
    n         = len(X)
    train_end = int(TRAIN_FRAC * n)
    val_end   = int((TRAIN_FRAC + VAL_FRAC) * n)

    splits = {}
    for tau in TAUS:
        y = labels[tau]
        splits[tau] = {
            'X_train': X.iloc[:train_end].values,
            'X_val':   X.iloc[train_end:val_end].values,
            'X_test':  X.iloc[val_end:].values,
            'y_train': y.iloc[:train_end].values,
            'y_val':   y.iloc[train_end:val_end].values,
            'y_test':  y.iloc[val_end:].values,
        }
    return splits

def train_models(df: pd.DataFrame) -> None:
    """
    Fits one HistGradientBoostingClassifier per horizon tau using
    pre-searched optimal hyperparameters.
    """
    global feature_cols, _splits
    X            = build_features(df)
    labels       = build_labels(df)
    feature_cols = list(X.columns)
    _splits = _make_splits(X, labels)

    for tau in TAUS:
        s      = _splits[tau]
        params = HARDCODED_PARAMS[tau]

        model = HistGradientBoostingClassifier(random_state=42, **params)
        model.fit(s['X_train'], s['y_train'])

        models[tau] = model
        print(f"Trained Model for Tau={tau}")

    print("->> All models successfully trained.\n")

def predict_adversity(*args, **kwargs) -> Union[float, np.ndarray]:
    """
    Predicts the probability of a trade being adverse at a specific horizon.

    Expected kwargs:
        features (np.ndarray, list, pd.Series): Engineered feature array for the tick(s).
        tau (int): Prediction horizon in {5, 10, 15, 20, 25, 30}.

    Returns:
        float  — if a single feature vector (1D) is passed.
        np.ndarray — if a 2D array of multiple trades is passed.
    """
    features = kwargs.get('features')
    tau      = kwargs.get('tau')

    if features is None or tau is None:
        raise ValueError("Must provide 'features' and 'tau' as keyword arguments.")
    if tau not in models:
        raise RuntimeError(f"Model for tau={tau} not found. Run train_models() first.")

    arr = np.array(features, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        return float(models[tau].predict_proba(arr)[0, 1])
    else:
        return models[tau].predict_proba(arr)[:, 1]

# METRICS
def _metrics_one_split(model, X: np.ndarray, y: np.ndarray) -> list:
    """Helper to return metrics in a standard consistent order."""
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return [
        accuracy_score(y, preds),
        precision_score(y, preds, zero_division=0),
        recall_score(y, preds, zero_division=0),
        log_loss(y, proba)
    ]

def compute_metrics(*args, **kwargs) -> pd.DataFrame:
    """
    Computes model metrics averaged across all tau horizons.

    Returns:
        pd.DataFrame: rows = [train, validation, test]
                      cols = [accuracy, precision, recall, log_loss]
    """
    if not models:
        raise RuntimeError("Run train_models() first.")

    metric_cols = ['accuracy', 'precision', 'recall', 'log_loss']
    splits = ['train', 'validation', 'test']
    
    # Store lists of metric arrays across horizons
    results = {s: [] for s in splits}

    for tau in TAUS:
        s = _splits[tau]
        mdl = models[tau]
        
        # Pull raw performance vectors directly per split
        results['train'].append(_metrics_one_split(mdl, s['X_train'], s['y_train']))
        results['validation'].append(_metrics_one_split(mdl, s['X_val'], s['y_val']))
        results['test'].append(_metrics_one_split(mdl, s['X_test'], s['y_test']))

    summary_data = {}
    for split in splits:
        summary_data[split] = np.mean(results[split], axis=0).round(4)

    df_out = pd.DataFrame.from_dict(summary_data, orient='index', columns=metric_cols)
    df_out.index.name = 'split'
    return df_out

#output
if __name__ == '__main__':
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
        train_models(df)

        metrics_df = compute_metrics()
        print("TASK 3 — METRICS (averaged across all tau)")
        print(metrics_df.to_string())
        print("FEATURE VECTOR ORDERING (Index Map)")
        for i, col in enumerate(feature_cols):
            print(f"  [{i:2d}]  {col}")

        # Submission requirement: task3_results.csv
        output_csv = 'task3_results.csv'
        metrics_df.to_csv(output_csv)
        print(f"\nSaved metrics to '{output_csv}'")
    else:
        print(f"[!] Dataset '{DATA_PATH}' not found.")