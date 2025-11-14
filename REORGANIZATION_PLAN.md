# Project Reorganization Plan

## Current Issues:
1. ML output files scattered in root directory
2. Multiple simulation result files in different locations
3. Cache data mixed with source code
4. Portfolio simulators not clearly organized

## Proposed New Structure:

```
Stock predictor app/
├── README.md                          # Main project README
├── .gitignore                         # Git ignore file
│
├── data/                              # All data files
│   ├── raw/                          # Original trade data
│   │   └── parsed_trade_entries_initial.json
│   ├── processed/                    # ML-labeled datasets
│   │   ├── ml_3R_labels.jsonl
│   │   ├── ml_5R_labels.jsonl
│   │   ├── ml_5R_labels_filtered.jsonl
│   │   └── summaries/
│   │       ├── ml_3R_summary.json
│   │       └── ml_5R_summary.json
│   └── cache/                        # Cached stock data
│       └── *.parquet files
│
├── src/                              # Source code
│   ├── data_processing/             # Data processing scripts
│   │   ├── Discord_Chat_Parser.py
│   │   └── Trade_Analyzer.py
│   │
│   ├── portfolio_simulation/        # Portfolio simulators
│   │   ├── __init__.py
│   │   ├── base_simulator.py       # Base simulator class
│   │   ├── realistic_simulator.py  # Realistic constraints
│   │   └── filtered_simulator.py   # Excludes split errors
│   │
│   ├── ml_models/                   # Machine learning models
│   │   └── (future ML training scripts)
│   │
│   └── utils/                       # Utility functions
│       └── config.py
│
├── results/                         # Simulation and analysis results
│   ├── portfolio_simulations/
│   │   ├── realistic_portfolio_results.json
│   │   ├── portfolio_simulation_results.json
│   │   └── filtered_portfolio_results.json
│   └── logs/
│       ├── portfolio_sim.log
│       └── trade_analyzer.log
│
├── notebooks/                       # Jupyter notebooks for analysis
│   └── (analysis notebooks)
│
├── tests/                          # Unit tests
│   └── (test files)
│
├── legacy/                         # Old/archived code
│   ├── Learning phase/
│   ├── Personal stuff/
│   └── breakout_trade/
│
└── Portfolio_Tele_Bot/            # Separate Telegram bot project
    ├── MyPortfolioTracker.py
    ├── requirements.txt
    └── ...
```

## Migration Steps:

1. Create new directory structure
2. Move ML output files to data/processed/
3. Move simulation results to results/
4. Reorganize source code
5. Move cache to data/cache/
6. Archive old code to legacy/
7. Update import paths
8. Create proper README files
9. Update .gitignore

