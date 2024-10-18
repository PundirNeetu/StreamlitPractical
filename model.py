import  pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# Load the dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# #train the model
# def train_model(df):
#     features = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']
#     target = 'life Expectancy (IHME)'
#     X= df[features]
#     y= df[target]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)


# save file as pickle file
    # with open('random_forest_model.pkl', 'wb') as f:
    #     pickle.dump(model, f)
    # return model

#predict life expectancy
def predict_life_expectancy(model, input_data):
    return model.predict([input_data])

#feature importance
def plot_feature_importance(model, feature_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature Importances")
    #plt.bar(range(len(feature_name)), importances[indices], align="center")
    plt.barh(range(len(feature_name)), importances[indices], align="center")
    plt.yticks(range(len(feature_name)), [feature_name[i] for i in indices])  # Set y-tick labels
    plt.xlim([0, max(importances) + 0.05])  # Adjust x-axis limit for better visibility
    plt.xlabel("Importance")
    plt.ylabel("Features")
    # Display the plot in Streamlit
    st.pyplot(plt)
    plt.clf()  # Clear the plot for future use
    

