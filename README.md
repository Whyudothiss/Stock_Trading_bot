
# Stock Breakout ML Pipeline

This project predicts high-probability breakout trades in US stocks using a machine learning model built entirely on pre-entry features. The goal is to identify setups with the best risk/reward before a trade is taken, so the model can be used in real-time or for backtesting without any lookahead bias.

## What It Does
- **Finds Breakout Setups:** Scans stocks for technical patterns that suggest a strong breakout is likely.
- **Evaluates Consolidation:** Measures the quality of the consolidation period before the breakout (tightness, volatility, volume trend, position in range, days near highs).
- **Analyzes Breakout Day:** Looks at the opening gap, first candle strength, and volume surge to assess the breakout's quality.
- **Assesses Entry Quality:** Judges the timing and risk/reward of the entry, including how close the stop is and whether the entry is early or late.
- **Predicts Probability of Success:** Uses a trained ML model to estimate the probability that a trade will reach a 5R (5x risk) target before hitting the stop.

## How It Works (Strategy)
1. **Consolidation Phase:**
   - The model looks for stocks that have been consolidating tightly for at least 5-10 days, with low volatility and declining volume (a sign of "energy building").
   - It prefers consolidations near the highs of the range, as these are more likely to break out strongly.
2. **Breakout Day:**
   - On the day of the breakout, the model checks for a small opening gap (not too large), a strong first candle (large range, closes near high), and a surge in volume compared to the average.
   - It also checks if the day's high is set early (within the first 30 minutes), which can be a sign of a failed breakout.
3. **Entry Execution:**
   - The model evaluates the entry price, stop loss, and target, ensuring the risk is reasonable and the reward is at least 5x the risk.
   - It prefers entries that are not too late after the breakout and where the stop is not too far from the entry.
4. **Prediction:**
   - All features are calculated using only data available before or at the time of entry—no future data is used.
   - The ML model (can be tree-based or neural network) predicts the probability of hitting the 5R target.

## Why This Approach?
- **No Data Leakage:** All predictions are made with information available at entry, so the model can be used live or for honest backtests.
- **Volume & Price Action:** The strategy combines classic technical analysis (consolidation, breakout, volume) with machine learning for objective, data-driven trade selection.
- **Proven Results:** Backtests show high win rates and strong returns when following the model's top-ranked signals.

