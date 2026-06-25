# Nomura Global Markets (GM) Quantitative Assignment

This repository contains the solutions for the Nomura Global Markets quantitative assessment, divided into three main sections: Q1, Q2, and Q3. 

## Repository Structure

### 📁 `q1/` - Theoretical and Mathematical Concepts
Contains solutions and mathematical derivations for the theoretical questions.
- `q1.csv`: Multiple-choice answers.
- `math_equations.txt`: Mathematical derivations and equations used for the theoretical questions (discount factors, forward rates, par swap rates, sensitivities, etc.).
- `Question1.docx`: Problem statements and detailed answers.

### 📁 `q2/` - C++ Pricing Engine
A C++ pricing engine for Interest Rate Swaps. It supports curve building, interpolation, present value (PV) calculation, and risk (sensitivity) analysis.
- **Key Features:**
  - Interest Rate Swap (IRS) pricing.
  - Curve construction from Cash Deposits and Vanilla Swaps.
  - Linear and AQ (Quadratic) Interpolation methods.
  - Risk analysis (Jacobian of Discount Factors w.r.t market quotes).
- **Files:**
  - `pricing_engine.cpp`: The core C++ source code.
  - `Input.csv`: Market data input (tenors, cash rates, swap rates).
  - `Output.csv`: Engine results including discount factors, PV, par swap rates, and risks.
  - `PricingEngine_Documentation.docx`: Detailed documentation of the pricing engine architecture and mathematics.

#### How to run Q2
```bash
cd q2
g++ -O3 -std=c++17 pricing_engine.cpp -o pricing_engine
./pricing_engine
```
*Note: Make sure `Input.csv` is in the same directory. Results will be generated in `Output.csv`.*

### 📁 `q3/` - Quantitative Analysis & Market Making Strategy
A set of Python scripts implementing quantitative trading analysis, adversity prediction, and a quoting strategy backtest based on trade data.
- **`t1.py`**: Simulates the adversity profile of clients at multiple time horizons (tau) based on mid-price changes.
- **`t2.py`**: Computes the expected PnL per trade, classifies clients as 'profitable' or 'costly', and calculates the minimum half-spread (delta) required for expected aggregate PnL to be non-negative.
- **`t3.py`**: Adversity Prediction Model. Uses a rigorous `RandomizedSearchCV` over `TimeSeriesSplit` to train an optimal `HistGradientBoostingClassifier` for predicting adverse trades at various horizons.
- **`t4.py`**: Calculates the optimal externalisation threshold ($\theta$) to maximize PnL on validation and test sets, comparing global vs. client-specific thresholds. Generates PnL vs. Theta plots.
- **`t5.py`**: Core quoting strategy and simulation backtest. Utilizes the adversity model, realized volatility, and inventory constraints to generate dynamic bid/ask quotes and calculates the overall strategy score (Total PnL / Portfolio Sigma), fill percentages, and max drawdowns.

#### How to run Q3
```bash
cd q3
pip install pandas numpy scikit-learn matplotlib
python t1.py
python t2.py
python t3.py
python t4.py
python t5.py
```
*Note: The scripts expect a `trade_data.csv` dataset in the appropriate directory to function properly.*

---

## Requirements
- **C++ Compiler**: GCC/Clang with C++17 support (for Q2).
- **Python**: 3.8+ with `pandas`, `numpy`, `scikit-learn`, and `matplotlib` (for Q3).
