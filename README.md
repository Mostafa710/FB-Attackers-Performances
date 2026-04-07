# Football Players Performance Analysis

## Overview
This project analyzes the performance of football players in Europe's top 5 leagues (Premier League, La Liga, Serie A, Bundesliga, and Ligue 1) from the 2017-2024 seasons. It includes exploratory data analysis (EDA), a machine learning model for predicting Expected Goals (xG), and interactive visualizations built with Dash and Power BI.

The project combines data science techniques with modern web and BI tools to provide insights into player demographics, league comparisons, attacking efficiency, and predictive analytics for forward players.

## Features
- **Data Exploration**: Comprehensive analysis of player profiles, nationalities, age distributions, and league-specific metrics.
- **Machine Learning Model**: Linear Regression model to predict Expected Goals (xG) for attackers based on various performance metrics.
- **Interactive Dashboard**: Dash web application with multiple tabs for overview, league comparisons, attacker analysis, and an xG predictor tool.
- **Power BI Report**: Additional visualizations and insights presented in a Power BI file.
- **Data Quality Checks**: Validation for missing values, duplicates, and categorical data consistency.

## Project Structure
```
├── app.py                          # Main Dash application
├── data_exploration.ipynb          # Jupyter notebook for EDA
├── ML_model.ipynb                  # Jupyter notebook for ML model development
├── datasets/                       # Data files
│   ├── cleaned_2017-18.csv
│   ├── cleaned_2018-19.csv
│   ├── cleaned_2019-20.csv
│   ├── cleaned_2020-21.csv
│   ├── cleaned_2021-22.csv
│   ├── cleaned_2022-23.csv
│   ├── cleaned_2023-24.csv
│   ├── full_dataset.csv            # Combined dataset
│   └── preprocessed_data.csv       # Preprocessed data
├── Model/
│   └── lr_model.joblib             # Trained Linear Regression model
└── README.md                       # This file
```

## Data Sources
The dataset consists of player performance statistics from Europe's top 5 leagues across 7 seasons (2017-2024). Each season's data is provided in separate cleaned CSV files, which are combined into `full_dataset.csv` for analysis.

Key metrics include:
- Player demographics (age, position, nationality)
- Match statistics (matches played, minutes per match)
- Attacking metrics (goals, assists, shots, xG)
- Defensive metrics (tackles, aerial duels, possessions lost)
- Passing metrics (pass completion, progressive passes)

## Installation and Setup
1. **Prerequisites**:
   - Python 3.8+
   - Jupyter Notebook
   - Required Python packages: `pandas`, `numpy`, `scikit-learn`, `plotly`, `dash`, `joblib`

2. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn plotly dash joblib
   ```

3. **Run the Dash Application**:
   ```bash
   python app.py
   ```
   The app will be available at `http://127.0.0.1:8050/`

4. **Explore Notebooks**:
   - Open `data_exploration.ipynb` in Jupyter to run the EDA analysis
   - Open `ML_model.ipynb` to review or retrain the xG prediction model

## Usage
### Dash Dashboard
The Dash app provides four main tabs:
- **Overview**: Age distributions, position breakdowns, and nationality maps
- **Leagues**: Radar charts comparing league styles across key metrics
- **Attackers**: xG trends, top performers, and shot accuracy analysis
- **xG Predictor**: Interactive tool to predict Expected Goals based on player inputs

### Machine Learning Model
The Linear Regression model predicts xG for forward players using features like age, matches played, assists, shots, and aerial duel success. The model achieves an R² score of approximately 0.85 on test data.

### Power BI Report
Open the Power BI file (`.pbix`) in Power BI Desktop to explore additional visualizations and insights.

## Model Details
- **Algorithm**: Linear Regression
- **Target**: Expected Goals (xG)
- **Features**: Age, Matches Played, Avg Minutes per Match, Assists, Penalty Kicks Made, Assists p90, Possessions Lost, Total Shots, % Shots on Target, Shots p90, % Aerial Duels Won, Shot Creating Actions p90
- **Performance**: RMSE ≈ 0.15, R² ≈ 0.85
- **Training Data**: Forward players with sufficient minutes played

## Team Members and Contributions
- **Mostafa Mamdouh Mohamed**: Exploratory Data Analysis (EDA) and Dash application development
- **Marwan Tamer Mahmoud**: Exploratory Data Analysis (EDA) and Dash application development
- **Hagar Ahmed Ghazi**: Power BI report creation and visualizations