import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

def load_data():
    return pd.read_csv('./data/filtered_US_data.csv')

data = load_data()

data['$ Raised (mm)^2'] = data['$ Raised (mm)']**2

if 'USEPUINDXD' in data.columns:
    data['USEPUINDXD_diff'] = data['USEPUINDXD'].diff().fillna(0)
if 'DFF' in data.columns:
    data['DFF_diff'] = data['DFF'].diff().fillna(0)

st.sidebar.title("Tech Layoff Analysis")
st.sidebar.markdown("Navigate through various sections of the analysis.")
st.sidebar.empty()  # Add some space

st.sidebar.markdown("## 📌 Navigation")
page = st.sidebar.radio("", ["📊 Regression Analysis", "📋 Data Table", "📈 Scatter Plots"])


if page == "📊 Regression Analysis":
    predictors = ['$ Raised (mm)', '$ Raised (mm)^2'] + [col for col in data.columns if 'Stage_' in col or 'Industry_' in col]
    if 'USEPUINDXD_diff' in data.columns:
        predictors.append('USEPUINDXD_diff')
    if 'DFF_diff' in data.columns:
        predictors.append('DFF_diff')

    X = data[predictors]
    X = X.fillna(X.mean())  
    y = data['# Laid Off'].fillna(data['# Laid Off'].mean())
    X = X.replace([np.inf, -np.inf], 1e9)  
    X = sm.add_constant(X)  
    # Fit the regression model
    model = sm.OLS(y, X).fit()

    p_values = model.pvalues

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

elif page == "📋 Data Table":
    st.title("Data Table")
    st.dataframe(data)
elif page == "📈 Scatter Plots":
    st.title("Scatter Plots")


    st.subheader("Scatter plot for # Laid Off vs. $ Raised (mm)^2")
    st.markdown("""
    This scatter plot illustrates the relationship between the squared amount of money raised by tech companies and the number of layoffs.
    The squared term helps capture any non-linear relationship between funding and layoffs.
    For instance, as funding increases, layoffs might decrease at a diminishing rate.
    """)
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x='$ Raised (mm)^2', y='# Laid Off', data=data, ax=ax1)
    st.pyplot(fig1)

    if 'Stage_IPO' in data.columns:
        st.subheader("# Laid Off vs. Stage_IPO")
        st.markdown("""
        This scatter plot demonstrates the relationship between companies in the IPO stage and the number of layoffs.
        Companies at the IPO stage might have unique financial and operational challenges that influence their layoff decisions.
        """)
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x='Stage_IPO', y='# Laid Off', data=data, ax=ax2)
        st.pyplot(fig2)


    if 'Industry_Retail' in data.columns:
        st.subheader("# Laid Off vs. Industry_Retail")
        st.markdown("""
        This scatter plot showcases the relationship between the retail industry and the number of layoffs.
        The retail industry, with its own set of challenges and dynamics, might have different layoff patterns compared to other industries.
        """)
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x='Industry_Retail', y='# Laid Off', data=data, ax=ax3)
        st.pyplot(fig3)



# # Required imports
# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import statsmodels.api as sm

# # Load the data
# def load_data():
#     return pd.read_csv('./data/filtered_US_data.csv')

# data = load_data()

# # Enhanced sidebar
# st.sidebar.title("Tech Layoff Analysis")
# st.sidebar.markdown("Navigate through various sections of the analysis.")
# st.sidebar.empty()  # Add some space

# # Sidebar for navigation with emojis for better visual appeal
# st.sidebar.markdown("## 📌 Navigation")
# page = st.sidebar.radio("", ["📊 Regression Analysis", "📋 Data Table"])


# if page == "📊 Regression Analysis":
#     # Define predictors and target
#     X = data[['USEPUINDXD', 'DFF', '$ Raised (mm)'] + [col for col in data.columns if 'Stage_' in col]]
#     X = X.fillna(X.mean())  # Handle NaN values
#     y = data['# Laid Off'].fillna(data['# Laid Off'].mean())
#     X = X.replace([np.inf, -np.inf], 1e9)  # Handle infinite values
#     X = sm.add_constant(X)  # Add constant for intercept

#     # Fit the regression model
#     model = sm.OLS(y, X).fit()

#     # Display the results
#     st.title("Linear Regression Analysis")

#     # R squared values
#     st.subheader("R-squared values:")
#     st.write(f"R-squared: {model.rsquared}")
#     st.write(f"Adjusted R-squared: {model.rsquared_adj}")

#     # Significance
#     st.subheader("Regression Coefficients and Significance:")
#     coeff_table = pd.DataFrame(model.summary().tables[1].data)
#     coeff_table.columns = coeff_table.iloc[0]
#     coeff_table = coeff_table.drop(0)
#     st.dataframe(coeff_table)

#     # Plotting variables
#     for col in X.columns:
#         if col != "const":  # Skip the 'const' column
#             st.subheader(f"Plot for {col}")


#             # Description for each plot
#             if col == "USEPUINDXD":
#                 st.markdown("This scatter plot visualizes the relationship between the number of layoffs and the Economic Uncertainty Index. A higher index value might indicate higher economic uncertainty, which could correlate with layoffs.")
#             elif col == "DFF":
#                 st.markdown("This scatter plot showcases the correlation between the number of layoffs and the Federal Funds Rate. Changes in this rate can impact economic conditions, potentially influencing layoffs.")
#             elif col == "$ Raised (mm)":
#                 st.markdown("Here, we see the relationship between the amount of money raised by tech companies (in millions) and the number of layoffs. Companies with more funding might be expected to have fewer layoffs, but other factors can come into play.")
#             elif "Stage_" in col:
#                 stage_name = col.split("_")[1]
#                 st.markdown(f"This scatter plot represents the number of layoffs for companies in the **{stage_name}** stage. Different stages might face different challenges and financial pressures, affecting layoff decisions.")


#             fig, ax = plt.subplots()
#             sns.scatterplot(x=col, y=y, data=data, ax=ax)
#             st.pyplot(fig)

# elif page == "📋 Data Table":
#     st.title("Data Table")
#     st.dataframe(data)
