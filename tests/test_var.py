import sys
sys.path.append("../src")

from app import var

import pandas as pd
import pytest
from unittest.mock import Mock, patch
import sys

def test_load_data():
    """Test loading data from a CSV file."""
    # Use the actual path to your test CSV file.
    df = var.load_data("./test_data/test_var.csv")  
    expected_columns = ["observation_date","FEDFUNDS","CORESTICKM159SFRBATL","INDPRO","LAYOFFS","UNCERTAINTY"]  # Adjust based on your CSV structure.
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == expected_columns
    assert not df.empty

def test_prepare_raw_data():
    """Test preparing raw data."""
    # Create a sample DataFrame
    raw_data = pd.DataFrame({
        "observation_date": ["7/1/2011", "8/1/2011"],
        "value": [10, 20],
        "Unnamed: 2": [None, None]  # Simulate an unwanted column.
    })
    processed_data = var.prepare_raw_data(raw_data)
    assert isinstance(processed_data, pd.DataFrame)
    # Check if "observation_date" is now the index and in datetime format.
    assert processed_data.index.equals(pd.DatetimeIndex(["7/1/2011", "8/1/2011"]))
    assert "value" in processed_data.columns
    assert "Unnamed: 2" not in processed_data.columns
    # Verify the index frequency is set to "MS" (month start frequency)
    assert processed_data.index.freqstr == "MS"

def test_get_irf():
    """Test generating impulse response functions from a fitted VAR model."""
    # Mock for the fitted model
    mock_fitted_model = Mock()
    # Directly assign a mock or a value to the return value of the irf method
    mock_fitted_model.irf.return_value = "Expected IRF Result"

    # Assuming get_irf() calls irf(20) on the fitted model and returns its result
    result = var.get_irf(mock_fitted_model)

    # Check that irf was called correctly and verify the result
    mock_fitted_model.irf.assert_called_once_with(periods=20)
    assert result == "Expected IRF Result", "The function should return the expected IRF result."


def test_difference_variables():
    """Test differencing variables to ensure stationarity."""
    # Create a sample DataFrame
    raw_data = pd.DataFrame({
        "CORESTICKM159SFRBATL": [1, 2, 3, 4],
        "INDPRO": [100, 200, 300, 400],
        "FEDFUNDS": [1.5, 1.7, 1.8, 2.0],
        "UNCERTAINTY": [30, 40, 50, 60],
        "LAYOFFS": [1000, 1100, 1200, 1300]
    })
    differenced_data = var.difference_variables(raw_data)

    assert "INFLATION" in differenced_data.columns and "D_INDPRO" in differenced_data.columns, \
        "The function should create 'INFLATION' and 'D_INDPRO' columns."
    assert not differenced_data.isnull().values.any(), \
        "The function should drop rows with NaN values resulting from differencing."
    

@pytest.mark.parametrize("maxlags", [1, 2, 3])
def test_fit_var_model_and_select_lags(maxlags):
    """Test fitting a VAR model and selecting optimal lags."""
    # Create a sample DataFrame
    raw_data = pd.DataFrame({
        "D_INDPRO": [0.1, 0.2, 0.3],
        "INFLATION": [0.01, 0.02, 0.03],
        "FEDFUNDS": [1.5, 1.7, 1.8],
        "UNCERTAINTY": [30, 40, 50],
        "LAYOFFS": [1000, 1100, 1200]
    })

    # Setup the mock to have aic, hqic, and bic attributes
    fit_result_mock = Mock()
    fit_result_mock.aic = 1.0
    fit_result_mock.hqic = 2.0
    fit_result_mock.bic = 3.0

    with patch("app.var.VAR") as mock_var: 
        mock_model = Mock()
        mock_model.fit.return_value = fit_result_mock
        
        mock_var.return_value = mock_model

        fitted_model, results_df = var.fit_var_model_and_select_lags(raw_data, maxlags)

        assert isinstance(fitted_model, Mock), "The function should return a fitted model mock."
        assert isinstance(results_df, pd.DataFrame), "The function should return a DataFrame of AIC, HQIC, and BIC values."
        assert not results_df.empty, "The results DataFrame should not be empty."
        assert mock_model.fit.call_count == maxlags + 1, f"VAR model should fit for each lag up to maxlags and once more with the best lag, expected {maxlags + 1} calls."
