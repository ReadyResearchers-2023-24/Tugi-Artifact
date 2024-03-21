import streamlit as st
from statsmodels.tsa.stattools import adfuller


from display import plot_irfs, display_data_tables, perform_regression_analysis, print_adf_result, display_var_model_results
from var import load_data, prepare_raw_data, difference_variables, fit_var_model_and_select_lags

# This function must be here because it contains the IRF

def main():
    # Load the dataset for Linear-Regression
    data = load_data('src/data/filtered_US_data.csv')
    data['$ Raised (mm)^2'] = data['$ Raised (mm)']**2

    # Load the dataset for VAR

    var_data = prepare_raw_data(load_data('src/data/macro_seasonal_variables_data.csv'))
    data_df = difference_variables(var_data)
    fitted_model, results_df = fit_var_model_and_select_lags(data_df,12)
    irf = fitted_model.irf(periods=20)

    # Side bar configurations

    st.sidebar.title("Tech Layoff Analysis")
    st.sidebar.markdown("Navigate through various sections of the analysis.")
    st.sidebar.empty()

    st.sidebar.markdown("## 📌 Navigation")
    page = st.sidebar.radio("", ["📊 Vector Auto Regression (VAR)","📊 Regression Analysis", "📋 Data Tables", "📈 Impulse Response Functions"])

    # Individual pages

    if page == "📊 Regression Analysis":
        perform_regression_analysis(data)

    elif page == "📊 Vector Auto Regression (VAR)":
        # Augmented Dickey-Fuller tests for stationarity (before differencing)
        st.header("ADF Tests Before Differencing\n")
        for var in ['LAYOFFS', 'UNCERTAINTY', 'FEDFUNDS', 'CORESTICKM159SFRBATL', 'INDPRO']:
            result = adfuller(var_data[var], 1)
            print_adf_result(var, result)

        st.header("ADF Tests After Differencing\n")
        for var in ['INFLATION', 'D_INDPRO']:
            result = adfuller(data_df[var].dropna())
            print_adf_result(var, result)
    
        # Display results in Streamlit
        display_var_model_results(fitted_model, results_df)

    elif page == "📋 Data Tables":
        display_data_tables(data, var_data)

    elif page == "📈 Impulse Response Functions":

        st.title('Impulse response functions')
        st.write('This section shows the response of layoffs to shocks in various economic indicators.')
        independent_variables = ['INFLATION', 'D_INDPRO', 'FEDFUNDS', 'UNCERTAINTY', 'LAYOFFS']
        for independent_vars in independent_variables:
            plot_irfs(irf, independent_vars)

if __name__ == "__main__":
    main()
