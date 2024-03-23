# Tech Layoff Analysis Artifact 2023

## Table of Contents


* [Outline](#outline)
    * [Preface Introduction and Motivation](#preface-introduction-and-motivation)
    * [Preface Data Section](#preface-data-section)
    * [Preface Technical Details](#preface-technical-details)
    * [Preface Experimental Results](#preface-experimental-results)
    * [Preface Future Work](#preface-future-work)
* [How to run the streamlit web application](#how-to-run-the-streamlit-web-application)
    * [Go the official research website](#go-the-official-research-website)
    * [Access it through poetry and virtual environment](#access-it-through-poetry-and-virtual-environment)
    * [Access it from the DockerHub](#access-it-from-the-dockerhub)
- [Conclusion](#conclusion)

## Outline

This comprehensive analysis explores the dynamics of technology sector layoffs within the context of broader economic indicators. Given the challenging nature of the job market in 2023, particularly in the technology industry, my goal was to uncover the key economic factors influencing the current employment landscape. To achieve this, I have crafted a sophisticated STATA code that not only runs on Stata to confirm my hypotheses but also encompasses a Python Vector Auto Regression Streamlit website, offering a deeper dive into the interplay between economic trends and tech layoffs.

_Tuguldur Gantulga_
_Timothy P. Bianco, PhD_
_Gregory Kapfhammer, PhD_

_Fall 2023 - Spring 2024_

_The Business and Economics Department_
_Department of Computer and Information Science_
_Allegheny College, Meadville, PA 16335_

### Preface Introduction and Motivation

In 2023, with a staggering 263,180 software engineers laid off by 1,193 companies, as reported by Layoffs.fyi, I sought to uncover the economic factors driving this turmoil in the U.S. job market. To conduct a thorough economic analysis, I applied my computer science expertise. The hypotheses and macroeconomic variables I formulated are as follows:

- H1: An increase in the Federal Funds Rate leads to an increase in layoffs.
- H2: An increase in inflation leads to an increase in layoffs.
- H3: An increase in industrial production leads to a decrease in layoffs.
- H4: An increase in the Economic Uncertainty Index leads to an increase in layoffs.

### Preface Data Section

The dataset underpinning the Vector Autoregression (VAR) model in my study encompasses monthly data ranging from December 2000 to December 2023, forming the basis of a comprehensive time series analysis. Utilizing public macroeconomic datasets, such as the Federal Funds Rate, the U.S. Uncertainty Index, Industrial Production, the Number of Layoffs in the IT Sector, and Inflation, all sourced from the Federal Reserve Economic Data (FRED), I meticulously examined the intricate dynamics at play. In addition to these datasets, my research involved a detailed examination of basic trend data from Layoffs.fyi, which I analyzed using Tableau to gain further insights into the patterns and precipitants of tech sector layoffs.

### Preface Technical Details

I utilized these data to run my Vector Auto Regression (VAR) on Stata and to test my hypothesis, as mentioned above in the Introduction section, using my own VAR code written in Python. After developing my VAR scripts, I aimed to display my results through Streamlit and to plot the Impulse Response Function (IRF) graphs necessary for this research question. When working on the streamlit software side, I successfully integrated tools such as Docker, Statsmodels, Python, NumPy, Poetry, and functional practices to ensure my results and code were accurate.

### Preface Experimental Results:

Based on my VAR results and the Impulse response functions from both Stata and my Python statsmodels VAR, they concluded that:

They Proved:

- H3: An increase in industrial production leads to a decrease in layoffs.
- H4: An increase in the Economic Uncertainty Index leads to an increase in layoffs.

They disproved and disgareed with:

- H1: An increase in the Federal Funds Rate leads to an increase in layoffs.
- H2: An increase in inflation leads to an increase in layoffs.

### Preface Future Work

In future work for this senior project, I aim to refine the macroeconomic variables impacting layoffs, acknowledging the complexity of accurately predicting layoffs due to factors such as company structure, business demand, organizational changes, and product specifications. The SBIC and HQIC results indicated that a 2-month lag is optimal for the Vector Autoregression (VAR) model, prompting plans to run the VAR with this lag and evaluate any deviations in the Impulse Response Function (IRF) results from those based on a 12-month lag. Additionally, I plan to explore various predictive models, including neural networks and machine learning algorithms, to assess their effectiveness in generating distinct IRF narratives

## How to run the streamlit web application:


### Access it through Poetry!

1. First go here and install Poetry

```text
https://python-poetry.org/docs/#installing-with-the-official-installer
```

2. Clone repository
3. Go to root of the project
4. Run these commands:
```
Poetry shell
```
```
Poetry install
```
```
Poetry run streamlit run src/app.main.py
```
5. The application will be accessible at `http://localhost:8501` on your web browser.

### Access it from the DockerHub:

- First, go to Docker and Install Docker Desktop

- To run the `tugi-artifact` app, first pull the image from Docker Hub:

```bash
docker pull tuduun/tugi-artifact:v1.0.1
```

- After pulling the image, run the application by executing the following command:

```bash
docker run -p 8501:8501 tuduun/tugi-artifact:v1.0.1
```

- The application will be accessible at `http://localhost:8501` on your web browser.
