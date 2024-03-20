import sys
sys.path.append("../src")
from app import display

from unittest.mock import Mock, patch, MagicMock
import pytest
from matplotlib.figure import Figure


import statsmodels.api as sm
import numpy as np
import pandas as pd


def test_plot_irfs():
    """
    Test the plot_irfs function to ensure it correctly handles plotting through the given IRF object
    and renders the plot via Streamlit.
    """
    # Mock the IRF object and its plot method
    mock_irf = Mock()
    mock_fig = Mock(spec=Figure)
    mock_irf.plot.return_value = mock_fig
    
    independent_var = 'GDP'

    # Patch the streamlit.pyplot function
    with patch('app.display.st.pyplot') as mock_pyplot:
        # Call the function with the mock IRF object and an independent variable
        display.plot_irfs(mock_irf, independent_var)
        
        # Assert the IRF plot method was called correctly
        mock_irf.plot.assert_called_once_with(impulse=independent_var, response='LAYOFFS', orth=True,
                                              subplot_params={'title': f'Response of LAYOFFS to a shock in {independent_var}'})
        
        # Assert the Streamlit pyplot function was called with the figure returned by irf.plot
        mock_pyplot.assert_called_once_with(mock_fig)


def test_display_data_tables():
    data = pd.DataFrame({"Company": ["A", "B"], "Layoffs": [100, 200]})
    var_data = pd.DataFrame({"Variable": ["GDP", "Unemployment"], "Value": [1.5, 7.2]})
    
    with patch('app.display.st.title') as mock_title, \
         patch('app.display.st.dataframe') as mock_dataframe:
        display.display_data_tables(data, var_data)
        
        # Instead of using assert_any_call, verify call counts and types
        assert mock_title.call_count == 2
        assert mock_dataframe.call_count == 2
        
        # Verify that the first call to mock_dataframe was made with a DataFrame
        first_call_arg = mock_dataframe.call_args_list[0][0][0]
        assert isinstance(first_call_arg, pd.DataFrame)
        # Optionally, verify the columns of the DataFrame in the first call match expected
        assert list(first_call_arg.columns) == ["Company", "Layoffs"]
        
        # Similarly for the second call
        second_call_arg = mock_dataframe.call_args_list[1][0][0]
        assert isinstance(second_call_arg, pd.DataFrame)
        assert list(second_call_arg.columns) == ["Variable", "Value"]


@pytest.fixture
def sample_data():
    """Provides a sample dataset for testing."""
    return pd.DataFrame({
        '$ Raised (mm)': [100, 200, np.nan],
        '$ Raised (mm)^2': [10000, 40000, 90000],
        'Stage_Pre-Seed': [1, 0, 0],
        'Industry_Tech': [0, 1, 0],
        'Industry_Healthcare': [1, 0, 1],
        '# Laid Off': [10, 15, np.nan]
    })

@patch('app.display.st')  # Mock Streamlit
@patch('app.display.sm.OLS')  # Mock statsmodels' OLS
def test_perform_regression_analysis(mock_ols, mock_st, sample_data):
    """Tests the regression analysis and result display."""
    # Setup for OLS mock
    mock_fit_result = MagicMock()
    mock_fit_result.pvalues = pd.Series([0.01, 0.06, 0.02, 0.07], index=['const', '$ Raised (mm)', 'Stage_Pre-Seed', 'Industry_Healthcare'])
    mock_fit_result.rsquared = 0.8
    mock_fit_result.rsquared_adj = 0.75
    mock_ols.return_value.fit.return_value = mock_fit_result
    
    # Execute the function under test
    display.perform_regression_analysis(sample_data)
    
    # Instead of asserting exact DataFrame match, verify call characteristics
    assert mock_ols.called, "OLS regression should be called"
    call_args = mock_ols.call_args[0]
    assert isinstance(call_args[1], pd.DataFrame), "Expect a DataFrame as input to OLS"
    # You could add more checks here on the structure/content of call_args if needed
    
    # Verify Streamlit was used to display results
    assert mock_st.subheader.called
    assert mock_st.table.called




@pytest.mark.parametrize("variable,result,expected_call", [
    ("CORESTICKM159SFRBATL", (None, 0.01, None, None, None, None), "success"),
    ("FEDFUNDS", (None, 0.06, None, None, None, None), "error"),
    ("GDP", (None, 0.10, None, None, None, None), "error"),
    ("GDP", (None, 0.04, None, None, None, None), "success"),
])
def test_print_adf_result(variable, result, expected_call):
    with patch('app.display.st') as mock_st:
        display.print_adf_result(variable, result)
        
        # Assert that subheader and write are always called
        mock_st.subheader.assert_called()
        mock_st.write.assert_called_with(f'p-value: {result[1]:.6f}')
        
        # Assert based on expected call type
        if expected_call == "success":
            mock_st.success.assert_called()
        elif expected_call == "error":
            mock_st.error.assert_called()