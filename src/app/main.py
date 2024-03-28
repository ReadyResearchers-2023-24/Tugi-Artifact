import streamlit as st
from statsmodels.tsa.stattools import adfuller


from display import plot_irfs, display_data_tables, perform_regression_analysis, print_adf_result, display_var_model_results
from var import load_data, prepare_raw_data, difference_variables, fit_var_model_and_select_lags

# This function must be here because it contains the IRF

def main():
    # Load the dataset for Linear-Regression
    data = load_data('src/data/filtered_US_data.csv')
    data['$ Raised (mm)^2'] = data['$ Raised (mm)']**2


    description = ["Inflation on Layoffs graph illustrates how a positive shock in inflation affects the number of layoffs. We can see that a positive shock in inflation decreases the number of layoffs until it hits 0 in the 6th month. This does not seem to match the hypothesis that I initially came up with.",
                "Industrial production on Layoffs graph shows a positive shock in the industrial production decreases layoffs until it reached 0 in the 8th month. This result lines up with Hypothesis 2: An increase in Industrial Production decreases Layoffs. This makes economic sense because if there is more production, there is more workforce behind the produced goods.",
                "Federal Funds Rate (aka Interest Rates) on Layoffs graph tells how the federal funds rate affects the number of layoffs. According to the graph, we can see a positive shock in federal funds, layoffs decrease until the 3rd month. My hypothesis disagrees with this result because in economic theory, at times when interest rates are high, people save more and consume as little as possible. Moreover, jobs will not be created due to the businesses not borrowing any money with high-interest rates to do more projects. Thus, it is intuitive to think that higher interest rates will lead to more layoffs due to the slowing down of the economy and the increase in borrowing of money for businesses makes it expensive to hire employees. Thus, the VAR IRF does not support the economic theory.",
                "Economic Uncertainty Index on Layoffs graph illustrates how the U.S economic uncertainty index affects layoffs. The graph starts with a positive spike and is on the positive side until it hits 0 at around the 2nd period. This is in line with the economic theory that supports that businesses are cutting back on the workforce during economic challenging times. Moreover, after the 2nd period in the IRF graph, the response(layoffs) fluctuates above and below the zero line, which suggests the impact of layoffs changes over time. Thus, the VAR IRF result was consistent with the H4: Increase in Economic Uncertainty index increases Layoffs."
                "Layoffs on Layoffs does not mean that much."
                ]
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
        for i in range(len(independent_variables)-1):
            plot_irfs(irf, independent_variables[i], description[i])

if __name__ == "__main__":
    main()
