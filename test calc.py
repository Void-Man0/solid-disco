import _csv
import concurrent.futures
import csv
import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pandas as pd
import ta
import ta.momentum as momentum
import talib
import yfinance as yf



def get_signal(ticker: str) -> str:
    # Get data on the ticker
    ticker_data = yf.Ticker(ticker)
    # Get the historical prices
    ticker_df = ticker_data.history(period='ytd')
    rsi = ta.momentum.RSIIndicator(ticker_df['Close'], window=14).rsi()
    # Determine the buy/sell signal
    if rsi[-1] is None:
        signal = "Unknown"
    elif rsi[-1] < 10:
        signal = "90% Buy"
    elif rsi[-1] < 20:
        signal = "80% Buy"
    elif rsi[-1] < 30:
        signal = "70% Buy"
    elif rsi[-1] > 95:
        signal = "95% Sell"
    elif rsi[-1] > 80:
        signal = "80% Sell"
    elif rsi[-1] > 70:
        signal = "70% Sell"
    else:
        signal = "Hold"
        return signal


def get_current_price(ticker: yf.Ticker) -> float:
    """
    Get the current price of a stock.

    Parameters:
    ticker (yf.Ticker): The stock ticker symbol.

    Returns:
    float: The current price of the stock.
    """
    return ticker.info["regularMarketPrice"]


def get_stock_info(ticker: str) -> yf.Ticker:
    stock = yf.Ticker(ticker)
    return stock


def get_market_cap(ticker: yf.Ticker):
    stock_info = get_stock_info(ticker)
    if stock_info:
        return stock_info.info["marketCap"]
    else:
        pass


def get_stock_analysis(ticker_symbol=None) -> dict:
    """
    Returns a dictionary containing the analysis results of the specified stock ticker.
    If no ticker symbol is specified, returns the fair value analysis for the S&P 500 index.
    The analysis includes the current price, fair value, valuation status, and buy/sell signal.
    :param ticker_symbol: (optional) the ticker symbol of the stock to analyze
    :return: a dictionary containing the analysis results
    """

    if ticker_symbol is not None:
        # Handle case where ticker symbol is passed in
        # Perform analysis on specified stock ticker
        # Create dictionary for stock analysis and return it
        stock_data = {
            "current_price": get_current_price(ticker_symbol),  # replace with actual current price for the stock
            "fair_value": get_fair_value(ticker_symbol),
            "valuation_status": get_valuation_status(ticker_symbol),
            # replace with actual valuation status for the stock
            "signal": get_signal(ticker_symbol)  # replace with actual buy/sell signal for the stock
        }
        return stock_data
    else:
        # Handle case where no ticker symbol is passed in
        # Perform analysis on S&P 500 index
        get_index_analysis()
        # create dictionary for index analysis and return it
        index_data = {
            "current_price": 0,
            "fair_value": get_fair_value("INDEX"),
            "valuation_status": "N/A",
            "signal": "N/A"
        }
        return index_data


def get_free_cash_flow(ticker: yf.Ticker, years: int, revenue_growth: List[float], operating_margin: float,
                       tax_rate: float, capital_expenditure_percent: float, working_capital_percent: float) -> float:
    revenue = ticker.financials.loc["Total Revenue"][0]
    ebit_margin = operating_margin
    ebit = revenue * ebit_margin
    taxes = ebit * tax_rate
    nopat = ebit - taxes
    depreciation = ticker.financials.loc["Depreciation"][0]
    capital_expenditures = -(revenue * capital_expenditure_percent)
    working_capital = -(revenue * working_capital_percent)
    free_cash_flow = nopat + depreciation + capital_expenditures + working_capital
    for i in range(1, years):
        revenue *= 1 + revenue_growth[i - 1]
        ebit_margin = operating_margin
        ebit = revenue * ebit_margin
        taxes = ebit * tax_rate
        nopat = ebit - taxes
        depreciation *= 1 - (capital_expenditure_percent + working_capital_percent)
        capital_expenditures = -(revenue * capital_expenditure_percent)
        working_capital = -(revenue * working_capital_percent)
        free_cash_flow += (nopat + depreciation + capital_expenditures + working_capital) / (
                1 + ticker.info['beta']) ** i
    return free_cash_flow


def get_index_rate(indexSymbol: str):
    index = yf.Ticker(indexSymbol)
    return index.history(period="ytd").pct_change().mean()


def get_valuation_status(current_price: float, fair_value: float) -> str:
    if current_price < fair_value:
        return "Undervalued"
    elif current_price > fair_value:
        return "Overvalued"
    else:
        return "Fairly Valued"


def get_fair_value(ticker: yf.Ticker, indexSymbol: str = "^GSPC", years: int = 1, revenue_growth: List[float] = [0.05],
                   operating_margin: float = 0.15, tax_rate: float = 0.25, capital_expenditure_percent: float = 0.1,
                   working_capital_percent: float = 0.2) -> float:
    try:
        index_rate = get_index_rate(indexSymbol)
        free_cash_flow = get_free_cash_flow(ticker, years, revenue_growth, operating_margin, tax_rate,
                                            capital_expenditure_percent, working_capital_percent)
        fair_value = free_cash_flow / ((1 + index_rate) ** years)
        return fair_value
    except:
        ticker_df = ticker.history(period='ytd')
        rsi = talib.RSI(ticker_df['Close'])
        fair_value = ticker_df['Close'].mean() * (1 + (rsi[-1] / 100 - 0.5) * 0.2)
        return fair_value


def get_index_analysis(indexSymbol: str) -> dict:
    """
    Returns a dictionary containing the analysis results of the specified index ticker.
    The analysis includes the current price, fair value, valuation status, and buy/sell signal.
    :param indexSymbol: the ticker symbol of the index to analyze
    :return: a dictionary containing the analysis results
    """
    # Get data on the index
    indexData = yf.Ticker('^' + indexSymbol)

    # Get the historical prices
    indexDf = indexData.history(period='ytd')

    # Calculate the RSI (Relative Strength Index)
    rsi = ta.momentum.RSIIndicator(indexDf['Close'], window=14).rsi()

    # Calculate the fair value and current price
    fair_value = indexDf['Close'].mean() * (1 + (rsi[-1] / 100 - 0.5) * 0.2)
    current_price = indexDf['Close'][-1]

    # Determine the valuation status
    if current_price > fair_value:
        valuation_status = "Overvalued"
    elif current_price < fair_value:
        valuation_status = "Undervalued"
    else:
        valuation_status = "Fairly valued"

    # Get data on the ticker
    ticker_data = yf.Ticker(ticker)
    # Get the historical prices
    ticker_df = ticker_data.history(period='ytd')
    # Calculate the RSI (Relative Strength Index)
    rsi = ta.momentum.RSIIndicator(ticker_df['Close'], window=14).rsi()
    # Determine the buy/sell signal
    if rsi[-1] is None:
        signal = "Unknown"
    elif rsi[-1] < 20:
        signal = "90% Buy"
    elif rsi[-1] < 30:
        signal = "80% Buy"
    elif rsi[-1] < 35:
        signal = "70% Buy"
    elif rsi[-1] > 90:
        signal = "95% Sell"
    elif rsi[-1] > 75:
        signal = "80% Sell"
    elif rsi[-1] > 65:
        signal = "70% Sell"
    else:
        signal = "Hold"

    # Create a dictionary to store the analysis results
    analysis_dict = {
        "current_price": current_price,
        "fair_value": fair_value,
        "valuation_status": valuation_status,
        "signal": signal,
    }

    return analysis_dict


def get_stock_analysis(ticker_symbol: str) -> dict:
    """
    Returns a dictionary containing the analysis results of the specified stock ticker.
    The analysis includes the current price, fair value, valuation status, and buy/sell signal.
    :param ticker_symbol: the ticker symbol of the stock to analyze
    :return: a dictionary containing the analysis results
    """

    # Get data on the ticker
    ticker_data = yf.Ticker(ticker_symbol)

    # Get the historical prices
    ticker_df = ticker_data.history(period='ytd')

    # Check if the DataFrame is empty
    if ticker_df.empty:
        return {"error": f"No data available for ticker symbol {ticker_symbol}"}

    # Calculate the RSI (Relative Strength Index)
    rsi = ta.momentum.RSIIndicator(ticker_df['Close'], window=14).rsi()

    # Calculate the fair value and current price
    fair_value = ticker_df['Close'].mean() * (1 + (rsi[-1] / 100 - 0.5) * 0.2)

    # Check if the DataFrame has at least one row
    if ticker_df.index.size > 0:
        current_price = ticker_df['Close'][-1]
    else:
        current_price = None

    # Determine the valuation status
    if current_price is None or fair_value is None:
        valuation_status = "Unknown"
    elif current_price > fair_value:
        valuation_status = "Overvalued"
    elif current_price < fair_value:
        valuation_status = "Undervalued"
    else:
        valuation_status = "Fairly valued"

    # Calculate the RSI (Relative Strength Index)
    rsi = ta.momentum.RSIIndicator(ticker_df['Close'], window=14).rsi()
    # Determine the buy/sell signal
    if rsi[-1] is None:
        signal = "Unknown"
    elif rsi[-1] < 20:
        signal = "90% Buy"
    elif rsi[-1] < 30:
        signal = "80% Buy"
    elif rsi[-1] < 35:
        signal = "70% Buy"
    elif rsi[-1] > 90:
        signal = "95% Sell"
    elif rsi[-1] > 75:
        signal = "80% Sell"
    elif rsi[-1] > 65:
        signal = "70% Sell"
    else:
        signal = "Hold"

    # Create a dictionary to store the analysis results
    analysis_dict = {
        "current_price": current_price,
        "fair_value": fair_value,
        "valuation_status": valuation_status,
        "signal": signal,
    }

    return analysis_dict


if __name__ == '__main__':
    # List of stock tickers to analyze
    stocks = ["AAPL", "GOOG", "MSFT", "AMD", "GME", "TGT", "BBY", "NVDA", "CRSR", "AMD", "TSLA", "AI", "AMZN", "SCHW",
              "INTC", "AMC", "PLTR", "COST", "WAL", "SCLX", "SRNEQ", "TRKA", "JNJ", "SIRI", "PLUG", "DLO", "NIO", "LEVI", "GCTK", "CHWRF", "LYSFF", "LITE"]
    tickers = ["AAPL", "GOOG", "MSFT", "AMD", "GME", "TGT", "BBY", "NVDA", "CRSR", "AMD", "TSLA", "AI", "AMZN", "SCHW",
               "INTC", "AMC", "PLTR", "COST", "WAL", "SCLX", "SRNEQ", "TRKA", "JNJ", "SIRI", "PLUG", "DLO", "NIO", "LEVI", "GCTK", "CHWRF", "LYSFF", "LITE"]
    data = []
    # Create a thread pool executor with 10 threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks to the thread pool for each stock ticker
        if not data == 0:
            # run calculation for true value
            future_to_stock = {executor.submit(get_stock_analysis, ticker): ticker for ticker in stocks}
        else:
            get_index_analysis()
            # run calculation for false value
            future_to_stock = {executor.submit(get_fair_value, yf.Ticker(ticker)): ticker for ticker in stocks}

        results = []

        for future in concurrent.futures.as_completed(future_to_stock):
            ticker = future_to_stock[future]
            stock_analysis = future.result()
            results.append((ticker, stock_analysis))

        for ticker in tickers:
            stock = yf.Ticker(ticker)
            current_price = stock.info['regularMarketPrice']
            fair_value = get_fair_value(yf.Ticker(ticker))
            valuation_status = get_valuation_status(current_price, fair_value)
            date = datetime.datetime.now().strftime('%Y-%m-%d')
            data.append([ticker, current_price, valuation_status, date])

        # Print the results
        for result in results:
            tickerSymbol = result[0]
            stock_data = result[1]
            print(f"Analysis for {tickerSymbol}:")
            print(f"\tCurrent price: ${stock_data['current_price']:.2f}")
            print(f"\tFair value: ${stock_data['fair_value']:.2f}")
            print(f"\tValuation status: {stock_data['valuation_status']}")
            print(f"\tBuy/Sell signal: {stock_data['signal']}")
        # Opens the file and writes ticker data
        with open("results.csv", "a", newline='') as file:
            writer: _csv.writer = csv.writer(file)
            writer.writerows(data)

        # create a DataFrame from the list of lists and write it to a CSV file
        df = pd.DataFrame(data, columns=["Ticker", "Current Price", "Valuation Status", "Last Updated"])
        df.to_csv("results_dataframe.csv", mode='a', header=not os.path.exists('results_dataframe.csv'), index=False)


