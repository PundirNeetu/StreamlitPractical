# train_model.py
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Train the model
def train_model(df):
    features = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']
    target = 'Life Expectancy (IHME)'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model as a pickle file
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved as 'random_forest_model.pkl'.")
    # Calculate MSE
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    print("Model trained and saved as 'random_forest_model.pkl'.")
    return mse
# Specify the path to your dataset
file_path = 'global_development_data.csv'  # Update with your actual file path
df = load_data(file_path)
mse_value = train_model(df) 