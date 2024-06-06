# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, request, jsonify
import pickle

# 1. Data Collection and Preprocessing

# Load dataset from CSV
data = pd.read_csv('sdss_data.csv')

# Preprocessing: Handle missing values, if any
data = data.dropna()

# Extract relevant features and target
features = data[['u', 'g', 'r', 'i', 'z']]
target = data['redshift']

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# 2. Exploratory Data Analysis (EDA)

# Histogram of redshift
plt.hist(data['redshift'], bins=30, edgecolor='k')
plt.title('Redshift Distribution')
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.show()

# Pairplot of features
sns.pairplot(data[['u', 'g', 'r', 'i', 'z']])
plt.show()

# Correlation matrix
corr_matrix = data[['u', 'g', 'r', 'i', 'z', 'redshift']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# 3. Predictive Modeling

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# 4. Visualization

# Scatter plot of actual vs predicted redshifts
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('Actual Redshift')
plt.ylabel('Predicted Redshift')
plt.title('Actual vs Predicted Redshift')
plt.show()

# Feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = ['u', 'g', 'r', 'i', 'z']

plt.figure()
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
plt.show()

# 5. Deployment (Flask Application)

app = Flask(__name__)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)