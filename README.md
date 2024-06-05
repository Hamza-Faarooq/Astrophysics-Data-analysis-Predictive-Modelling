
# Astrophysical Data Analysis and Predictive Modeling

This project involves analyzing astrophysical data from the Sloan Digital Sky Survey (SDSS) and building a predictive model to estimate the redshift of celestial objects. The project demonstrates skills in data processing, machine learning, data visualization, and web deployment using Flask.

## Project Structure

- `generate_mock_data.py`: Script to create and save a mock SDSS dataset as `sdss_data.csv`.
- `astro_project.py`: Main script that includes data collection, preprocessing, EDA, feature engineering, predictive modeling, evaluation, visualization, and deployment using Flask.
- `requirements.txt`: List of required Python packages.

 

### Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)


## Usage

### Data Collection and Preprocessing

The script loads the `sdss_data.csv` file, preprocesses it by handling missing values, and extracts relevant features and the target variable.

### Exploratory Data Analysis (EDA)

The script performs EDA, including:
- Histogram of redshift distribution
- Pairplot of features
- Correlation matrix heatmap

### Feature Engineering

The script standardizes the features using `StandardScaler` and splits the data into training and testing sets.

### Predictive Modeling

The script builds a `RandomForestRegressor` model to predict redshift values, trains the model, and evaluates its performance using Mean Squared Error (MSE) and R² score.

### Visualization

The script visualizes:
- Actual vs. Predicted redshift scatter plot
- Feature importances bar plot

### Deployment (Flask Application)

The script includes a simple Flask application that serves the trained model and allows users to make predictions via a REST API.

To run the Flask application, execute:
```bash
python astro_project.py
