import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import metrics


def get_stock_data(ticker_symbol, start_date, end_date):
    """
    Retrieve historical stock data from Yahoo Finance API.

    Parameters:
    - ticker_symbol: Ticker symbol of the stock (e.g., "601988.SS").
    - start_date: Start date in the format "YYYY-MM-DD".
    - end_date: End date in the format "YYYY-MM-DD".

    Returns:
    - DataFrame containing the historical stock data with 'Date' as the index
      and frequency set to 'D' (daily).
    """
    # Download historical stock data
    df = yf.download(ticker_symbol, start=start_date, end=end_date)
        
    return df

# Example usage:
# stock_data = get_stock_data("AAPL", "2001-01-01", "2022-01-01")


def perform_adf_test(data):
    """
    Perform Augmented Dickey-Fuller test to check for stationarity in time series data.

    Parameters:
    - data: Time series data (1D array, Series, or DataFrame column).

    Returns:
    - None

    This function performs the Augmented Dickey-Fuller (ADF) test on the input time series data
    to determine its stationarity. It prints the ADF test results, including the ADF statistic,
    p-value, and critical values. It also provides an interpretation of the test results,
    indicating whether the data is stationary or not based on a significance level of 0.05.
    """
    # Perform Augmented Dickey-Fuller test
    result = sm.tsa.adfuller(data)
    
    # Print ADF test results
    print("ADF Test Results:")
    print(f"ADF Statistic: {result[0]}")
    print(f"P-value: {result[1]}")
    print(f"Lags Used: {result[2]}")
    print(f"Number of Observations Used: {result[3]}")
    print("\nCritical Values:")
    
    # Print critical values
    for key, value in result[4].items():
        print(f"\t{key}: {value}")

    # Interpret the results
    if result[1] <= 0.05:
        print("\nReject the null hypothesis. The data is stationary.")
    else:
        print("\nFail to reject the null hypothesis. The data is not stationary.")

# Example usage:
# Assuming 'data' is a pandas Series or DataFrame column containing your time series data
# perform_adf_test(data)


def plot_acf_pacf(data, acf_lags=40, pacf_lags=40, figsize=(10, 8)):
    """
    Plot ACF and PACF for a given time series data.

    Parameters:
    - data: Time series data (1D array, Series, or DataFrame column).
    - acf_lags: Number of lags to include in the ACF plot. Default is 40.
    - pacf_lags: Number of lags to include in the PACF plot. Default is 40.
    - figsize: Size of the figure (width, height). Default is (10, 8).

    Returns:
    - None (displays the plots).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot ACF
    sm.graphics.tsa.plot_acf(data, lags=acf_lags, ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF)')
    ax1.set_xlabel("Correlation Coefficient")

    # Plot PACF
    sm.graphics.tsa.plot_pacf(data, lags=pacf_lags, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    ax2.set_xlabel("Correlation Coefficient")

    # Display the plots
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming 'data' is a pandas Series or DataFrame column containing your time series data
# plot_acf_pacf(data, acf_lags=30, pacf_lags=20, figsize=(12, 6))


def evaluation_metric(ytest, yhat, return_dict=False):
    """
    Calculate and print evaluation metrics for regression models.

    Parameters:
    - ytest: True target values (ground truth).
    - yhat: Predicted target values.
    - return_dict: If True, return metrics as a dictionary; otherwise, print them.

    Returns:
    - Dictionary containing evaluation metrics (if return_dict=True).
    """
    MSE = metrics.mean_squared_error(ytest, yhat)
    RMSE = MSE**0.5
    MAE = metrics.mean_absolute_error(ytest, yhat)
    R2 = metrics.r2_score(ytest, yhat)

    metrics_dict = {
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE,
        'R2': R2
    }

    if return_dict:
        return metrics_dict
    else:
        print('MSE: %.5f' % MSE)
        print('RMSE: %.5f' % RMSE)
        print('MAE: %.5f' % MAE)
        print('R2: %.5f' % R2)

# Example usage:
# metrics_result = evaluation_metric(y_test, y_pred, return_dict=True)



