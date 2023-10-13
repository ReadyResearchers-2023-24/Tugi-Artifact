# Required imports
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Load the data
@st.cache
def load_data():
    return pd.read_csv('./data/filtered_US_data.csv')

data = load_data()

# Define predictors and target
X = data[['USEPUINDXD', 'DFF', '$ Raised (mm)'] + [col for col in data.columns if 'Stage_' in col]]

# Handle NaN values
X = X.fillna(X.mean())
y = data['# Laid Off'].fillna(data['# Laid Off'].mean())

# Handle infinite values
X = X.replace([np.inf, -np.inf], 1e9)

# Add constant for intercept
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Display the results
st.title("Linear Regression Analysis")

# R squared values
st.subheader("R-squared values:")
st.write(f"R-squared: {model.rsquared}")
st.write(f"Adjusted R-squared: {model.rsquared_adj}")

# Significance
st.subheader("Regression Coefficients and Significance:")
# Convert the summary table to a DataFrame for better display in Streamlit
coeff_table = pd.DataFrame(model.summary().tables[1].data)
coeff_table.columns = coeff_table.iloc[0]
coeff_table = coeff_table.drop(0)
st.write(coeff_table)

# Plotting variables
for col in X.columns:
    if col != "const":  # Skip the 'const' column
        st.subheader(f"Plot for {col}")
        fig, ax = plt.subplots()
        sns.scatterplot(x=col, y=y, data=data, ax=ax)
        st.pyplot(fig)