from typing import Tuple
import pandas as pd
from statsmodels.tsa.api import VAR

def load_data(relative_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
        relative_path (str): The relative path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    return pd.read_csv(relative_path)

def prepare_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares raw data for analysis by converting dates and setting the index.

    Parameters:
        data (pd.DataFrame): The raw data with an 'observation_date' column.

    Returns:
        pd.DataFrame: The processed data with a datetime index.
    """
    data['observation_date'] = pd.to_datetime(data['observation_date'],  format='%m/%d/%Y')
    data.set_index('observation_date', inplace=True)
    data.drop(columns=[col for col in data.columns if 'Unnamed' in col], inplace=True)
    data.index.freq = 'MS'
    return data

def get_irf(fitted_model):
    """
    Generates impulse response functions from a fitted VAR model.

    Parameters:
        fitted_model (VARResultsWrapper): The fitted VAR model.

    Returns:
        VARResultsWrapper: The impulse response functions.
    """
    irf = fitted_model.irf(periods=20)
    return irf

def difference_variables(data: pd.DataFrame) -> pd.DataFrame:
    """
    Differences specific variables to ensure stationarity.

    Parameters:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The data with differenced variables.
    """
    data['INFLATION'] = data['CORESTICKM159SFRBATL'].diff()
    data['D_INDPRO'] = data['INDPRO'].diff()
    var_df = data[['D_INDPRO', 'INFLATION', 'FEDFUNDS', 'UNCERTAINTY', 'LAYOFFS']].dropna()
    return var_df

def fit_var_model_and_select_lags(data: pd.DataFrame, maxlags: int) -> Tuple[VAR, pd.DataFrame]:
    """
    Fits a VAR model and selects the optimal number of lags based on AIC.

    Parameters:
        data (pd.DataFrame): The input data for the VAR model.
        maxlags (int): The maximum number of lags to consider.

    Returns:
        Tuple[VARResultsWrapper, pd.DataFrame]: The fitted VAR model and a DataFrame with AIC, HQIC, and BIC for each lag.
    """
    model = VAR(data)
    results_aic = []
    for i in range(1, maxlags + 1):
        result = model.fit(i, trend='c')
        results_aic.append({'Lag': i, 'AIC': result.aic, 'HQIC': result.hqic, 'BIC': result.bic})
    results_df = pd.DataFrame(results_aic).set_index('Lag')
    best_lag = results_df['AIC'].idxmin()
    fitted_model = model.fit(best_lag)
    return fitted_model, results_df
