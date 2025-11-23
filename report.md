# Project Report: Data Source Evolution for Intraday Stock Prediction

## Introduction
This report documents the evolution of data sourcing strategies for my intraday stock prediction project. The goal was to obtain reliable, high-frequency (1-min and 5-min) stock data for model training and backtesting.

## Initial Approach: yfinance
I initially used the `yfinance` library to fetch daily and intraday stock data. While yfinance is convenient and free, it has a significant limitation: intraday data is only available for the past 60 days. This restriction made it unsuitable for building models that require longer historical intraday data.

## Transition to Alpaca API
To overcome the 60-day limit, I switched to the Alpaca API, which offers free access to historical intraday data (1-min and 5-min bars) for US stocks. This allowed me to collect a much larger dataset for backtesting and model training.

## Data Quality Issues with Alpaca
After running several backtests using Alpaca data, I noticed inconsistencies and potential issues in the results. To investigate, I compared Alpaca's intraday data with yfinance's data for overlapping periods. The comparison revealed significant discrepancies between the two sources, raising concerns about the reliability of Alpaca's intraday data for my use case.

## Migration to Twelve Data
Given these issues, I explored alternative data providers and found Twelve Data, which also offers free access to historical intraday data. I implemented a data comparison script to verify the similarity of Twelve Data's intraday bars with those from yfinance. The results showed that Twelve Data's data closely matches yfinance for most tickers and periods, providing greater confidence in its accuracy.

## Conclusion
The data source for this project evolved from yfinance (limited by 60-day intraday history), to Alpaca (unlimited but with quality concerns), and finally to Twelve Data (free, reliable, and with sufficient historical coverage). This process highlights the importance of validating data quality and consistency when building financial models.

---

*For more details on the data comparison methodology and scripts, see the `src/ml_pipeline/data_source_comparator.py` file in this repository.*
