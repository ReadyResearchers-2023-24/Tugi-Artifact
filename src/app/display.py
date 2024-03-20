import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
import pandas as pd

def plot_irfs(irf, independent_var):
    fig = irf.plot(impulse=independent_var, response='LAYOFFS', orth=True, subplot_params={'title': f'Response of LAYOFFS to a shock in {independent_var}'})
    st.pyplot(fig)

def display_data_tables(data, var_data):
    st.title("Layoffs.fyi Company Table:")
    st.dataframe(data)
    st.title("VAR macrovariables used for regression (Dec 2000 to Dec 2023):")
    st.dataframe(var_data)


def perform_regression_analysis(data):
    predictors = ['$ Raised (mm)', '$ Raised (mm)^2'] + [col for col in data.columns if 'Stage_' in col or 'Industry_' in col]
    # Add other predictors if present
    for col in ['USEPUINDXD_diff', 'DFF_diff']:
        if col in data.columns:
            predictors.append(col)

    X = data[predictors].fillna(data[predictors].mean()).replace([np.inf, -np.inf], 1e9)
    y = data['# Laid Off'].fillna(data['# Laid Off'].mean())
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Process and display results
    # We can separate here
    p_values = model.pvalues


    significant_predictors = p_values[p_values < 0.05].reset_index()
    significant_predictors.columns = ['Predictor', 'P-Value']


    st.subheader("Significant Predictors (p < 0.05):")
    st.table(significant_predictors)

    non_significant_predictors = p_values[p_values > 0.05].reset_index()
    non_significant_predictors.columns = ['Predictor', 'P-Value']


    st.title("Linear Regression Analysis")

    st.subheader("R-squared values:")
    st.write(f"R-squared: {model.rsquared}")
    st.write(f"Adjusted R-squared: {model.rsquared_adj}")

    st.subheader("Significant Predictors (p < 0.05):")
    st.table(significant_predictors)

    st.subheader("Non-Significant Predictors (p >= 0.05):")
    st.table(non_significant_predictors)


def print_adf_result(variable, result):
    if variable == "CORESTICKM159SFRBATL":
        st.subheader(f'Augmented Dickey-Fuller Test on "{variable}" aka "INFLATION":')
    else:
        st.subheader(f'Augmented Dickey-Fuller Test on "{variable}":')
    st.write(f'p-value: {result[1]:.6f}')
    if result[1] <= 0.05:
        st.success(f'"{variable}" is stationary at 5% significance level.\n')
    elif(variable == 'FEDFUNDS'):
        st.error(f'"{variable}" is not stationary, but we will not difference the {variable}.\n')
    else:
        st.error(f'"{variable}" is not stationary and requires differencing.\n')


def display_var_model_results(fitted_model, results_df):
    # Display Optimal Lag Lengths based on Information Criteria
    st.write("### Optimal Lag Lengths based on Information Criteria:")
    st.dataframe(results_df)

    # VAR Model Summary (General)
    st.write("### VAR Model Summary:")
    st.text(str(fitted_model.summary()))

    stats_list = []

    # Equation-specific statistics (RMSE, R-squared, etc.)
    st.write("### Equation-specific statistics:")
    resid = fitted_model.resid
    endog = fitted_model.endog
    sigma_u = fitted_model.sigma_u

    for i, eq_name in enumerate(fitted_model.model.endog_names):
        rmse = np.sqrt(np.diag(sigma_u))[i]
        # Adjust access to actuals and residuals
        actuals = fitted_model.endog[12:, i] if isinstance(fitted_model.endog, np.ndarray) else fitted_model.endog.iloc[12:, i].to_numpy()
        resid = fitted_model.resid[:, i] if isinstance(fitted_model.resid, np.ndarray) else fitted_model.resid.iloc[:, i].to_numpy()
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((actuals - np.mean(actuals))**2)
        r_squared = 1 - ss_res / ss_tot

        stats_list.append({"Equation": eq_name, "RMSE": f"{rmse:.6f}", "R-squared": f"{r_squared:.4f}"})
    ## additional results table
    stats_df = pd.DataFrame(stats_list) 

    # Convert DataFrame to Markdown table string
    md_table = stats_df.to_markdown(index=False)

    # Display the Markdown table in Streamlit
    st.markdown(md_table, unsafe_allow_html=True)