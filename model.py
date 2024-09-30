import random
import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import requests
import joblib
from config import data_base_path, model_file_path, TOKEN

def get_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['history']
        df = pd.DataFrame(data)
        df['t'] = pd.to_datetime(df['t'], unit='s')
        df.columns = ['Timestamp', 'Predict']
        df['Predict'] = df['Predict'].apply(lambda x: x * 100)
        print(df.head())
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
    return df

def download_data(token):
    os.makedirs(data_base_path, exist_ok=True)
    if token == 'R':
        url = "https://clob.polymarket.com/prices-history?interval=all&market=21742633143463906290569050155826241533067272736897614950488156847949938836455&fidelity=720"
        data = get_data(url)
        save_path = os.path.join(data_base_path, 'polymarket_R.csv')
        data.to_csv(save_path)
    elif token == 'D':
        url = "https://clob.polymarket.com/prices-history?interval=all&market=69236923620077691027083946871148646972011131466059644796654161903044970987404&fidelity=720"
        data = get_data(url)
        save_path = os.path.join(data_base_path, 'polymarket_D.csv')
        data.to_csv(save_path)

def train_model(token):
    if token == 'R':
        training_price_data_path = os.path.join(data_base_path, "polymarket_R.csv")
    elif token == 'D':
        training_price_data_path = os.path.join(data_base_path, "polymarket_D.csv")

    df = pd.read_csv(training_price_data_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df['hour'] = df['Timestamp'].dt.hour
    
    # Define features and target
    X = df[['year', 'month', 'day', 'hour']]
    y = df['Predict']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ridge regression with hyperparameter tuning (alpha)
    ridge = Ridge()
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best alpha: {grid_search.best_params_}")

    # Make predictions
    y_pred = best_model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Save the model
    os.makedirs(model_file_path, exist_ok=True)
    if token == 'R':
        save_path_model = os.path.join(model_file_path, 'ridge_model_R.pkl')
    elif token == 'D':
        save_path_model = os.path.join(model_file_path, 'ridge_model_D.pkl')

    joblib.dump(best_model, save_path_model)
    print(f"Trained model saved to {save_path_model}")

def get_inference(token):
    if token == 'R':
        save_path_model = os.path.join(model_file_path, 'ridge_model_R.pkl')
    elif token == 'D':
        save_path_model = os.path.join(model_file_path, 'ridge_model_D.pkl')

    loaded_model = joblib.load(save_path_model)
    print("Loaded model successfully")

    single_input = pd.DataFrame({
        'year': [2024],
        'month': [10],
        'day': [10],
        'hour': [12]
    })

    # Scaling single input
    scaler = StandardScaler()
    single_input_scaled = scaler.fit_transform(single_input)

    predicted_price = loaded_model.predict(single_input_scaled)
    return predicted_price[0]
    
