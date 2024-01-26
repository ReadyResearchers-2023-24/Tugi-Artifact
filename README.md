# Tech Layoff Analysis Artifact 2023

An in-depth analysis of the dynamics of tech layoffs in light of economic indicators/indicators.

## Introduction and Motivation

As the economy grapples with uncertainties, notably evident from the Economic Uncertainty Index and fluctuating federal funds rates, it's crucial to understand the underlying factors affecting tech layoffs. Through meticulous data analysis using Python, this project aims to unravel the correlations and potential causative factors. Harnessing the power of Streamlit, a web-based visualization tool, we aim to provide interactive insights that elucidate the patterns and trends in tech layoffs, offering stakeholders meaningful data to make informed decisions.

## Technical Details

- **Data Analysis**: Utilize Python for in-depth data analysis and trend extraction.
- **Data Source**: Aggregate data from multiple sources, including `Layoffsfyi Tracker`, Economic Uncertainty Index, and Federal Funds Rate FROM FRED.
- **Visualization**: Use Python's Streamlit library, a robust tool for web-based data visualization, to present the findings.
- **End Product**: An interactive web application to provide people analysis of the tech layoff landscape.

## Future Plans

- **Enhanced Data Integration**: Integrate more granular data points, like company-specific financial health metrics, and increase my R-squared value.
- **Modeling Improvements**: Explore advanced machine learning models to improve predictive capabilities concerning tech layoffs.
- **User-Centric Design**: Refine the Streamlit web application's UI/UX to ensure user-friendliness and ease of navigation.
- **Multimedia Integration**: Consider integrating video or animated presentations to explain complex findings more intuitively.

## Related Work

- **Tech Layoffs in Silicon Valley**: An analysis of the cyclic nature of tech layoffs in the heart of the tech industry.
- **Economic Indicators and their Impact on the Tech World**: A deep dive into how broad economic metrics directly and indirectly influence tech startups and established companies.
- **The Resilience of the Tech Industry in Economic Downturns**: Historical analysis of the tech industry's performance during past recessions.
- **Venture Capital and its Role in Tech Stability**: Understanding how investments play a role in both the boom and bust of tech startups.

## How to run DEMO and see the results:



```
cd src

python3 -m venv myvenv
source myvenv/bin/activate
pip install --upgrade pip
pip install streamlit
pip install pandas
pip install numpy
pip install seaborn
pip install matplotlib
pip install statsmodels

streamlit run streamlit.py
```
