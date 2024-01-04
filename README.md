
# Time-Series-Seasonal-Forecasting-with-Prophet
## Overview
This project is focused on forecasting time series data using the Prophet library, which is designed for making forecasts that are robust to shifts in trends and various seasonality patterns. The code demonstrates how to preprocess the data, train a Prophet model, and visualize the results.

## Structure
- `data/`: Directory containing training and validation datasets.
- `myenv/`: Virtual environment directory for the project.
- `myenv_prophet/`: Separate virtual environment for Prophet.
- `notebook/`: Jupyter notebooks for exploration and analysis.
- `scripts/`: Python scripts for executing the forecast model.
- `requirements.txt`: Required libraries for the project setup.

## Prerequisites
Before running the scripts, ensure you have the following:
- Python 3.7 or higher.
- Pip package manager.

## Installation
To install the required Python packages, run the following command:
```bash
pip install -r requirements.txt
```

## Running the Script
To execute the time series forecasting script, navigate to the `scripts/` directory and run:
```bash
python time_series_seasonal_forecasting_prophet.py
```

## Code Description
The script executes the following steps:
- Data loading and preprocessing.
- Time series visualization of training and validation data.
- Model fitting with Prophet.
- Future predictions and visualization of forecasts.
- Calculation of RMSE to evaluate model performance.

## Data Preprocessing
The training and validation data timestamps are converted to datetime objects, and the data is indexed based on these timestamps for time series analysis.

## Forecasting
A Prophet model is initialized, fitted on the training data, and used to make predictions on future data points. The predictions are then visualized alongside the actual data.

## Evaluation
The Root Mean Square Error (RMSE) is calculated between the validation data and the forecasts to assess the accuracy of the model.

## Visualization
The script generates several plots showing the actual vs. predicted values and the forecast components.

Remember to replace the paths to the CSV files in the script with the actual paths where your datasets are located.

## Contributing
Contributions to the project are welcome. Please fork the repository and open a pull request with your changes.

## License
Distributed under the MIT License. See `LICENSE` for more information.
```
