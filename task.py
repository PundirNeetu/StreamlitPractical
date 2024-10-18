import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pydeck as pdk
from plots import create_scatter_plot,create_3d_bar_plot, load_map_data
from model import predict_life_expectancy,plot_feature_importance
from train_model import load_data,train_model
from map import create_choropleth_map

#st.write("Hello, world!")
#st.write("There is more to it. just HOLD ON!!")
##Task 1
#write headline as header "Worldwide Analysis of Quality of Life and Economic Factors"
#write subtitle "This app enables you to explore the relationships between poverty, 
#            life expectancy, and GDP across various countries and years. 
#            Use the panels to select options and interact with the data."
#use the whole width of the page
#create 3 tabs called ""Global Overview", "Country Deep Dive", "Data Explorer"
st.set_page_config(page_title="Worldwide Analysis of Quality of Life and Economic Factors", layout="wide")

st.title("Worldwide Analysis of Quality of Life and Economic Factors")
st.markdown("This app enables you to explore the relationships between poverty, life expectancy, and GDP across various countries and years. Use the panels to select options and interact with the data.")
tabs = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])


#Task 2 #taks 2 in tab 3
#use global_development_data.csv which is a cleaned merge of those 3 datasets

#poverty_url = 'https://raw.githubusercontent.com/owid/poverty-data/main/datasets/pip_dataset.csv'
#life_exp_url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Healthy%20Life%20Expectancy%20-%20IHME/Healthy%20Life%20Expectancy%20-%20IHME.csv"
#gdp_url = 'https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Maddison%20Project%20Database%202020%20(Bolt%20and%20van%20Zanden%20(2020))/Maddison%20Project%20Database%202020%20(Bolt%20and%20van%20Zanden%20(2020)).csv'
    
#show the dataset in the 3rd tab
#read in the dataset and show it
#include a multiselectbox to select the country names
#include a slider to select the year range
#make the filtered dataset downloadable

with tabs[2]:
    #show the dataset in the 3rd tab
    #read in the dataset and show it
    st.subheader("Data Explorer")
    df = pd.read_csv('global_development_data.csv')
    st.dataframe(df.head())	
    
    # Create a multiselectbox for country selection
    countries = df['country'].unique()
    selected_countries = st.multiselect("Select countries:", countries)

    # Create a slider for year range
    # year are 1990 to 2016
    #year_range = st.slider('year range', min_value =1990, max_value=2016, value=(1990,2016))
    year_range = st.slider("Year Range", min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=(int(df['year'].min()), int(df['year'].max())))

    #filter the datset based on userinput
    filtered_data = df[(df['country'].isin(selected_countries)) & (df['year'].between(*year_range))]
    
    #make the filtered dataset downloadable
    st.write("### Filtered Dataset")
    st.dataframe(filtered_data)

    # Download filtered dataset
    if st.button("Download Filtered Dataset"):
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='filtered_data.csv',
            mime='text/csv'
        )
#task 3: deployment: deploy the app on streamlit cloud (see readme: create own github repo with practical.py file and requirements.txt, connect the github to streamlit cloud)
# Done
#task 4 in tab 1
#create a slider to select a certain year, filter the dataset accordingly
#create 4 key metrics in 4 columns each with a description: 
#col1: mean of life expectancy; 
#col2: median of GDP per capita; 
#col3: mean of headcount_ratio_upper_mid_income_povline; 
#col4: Number of countries

 #create a slider to select a certain year, filter the dataset accordingly
with tabs[0]:
    st.title("Global-Overview Key statistics")
    year_range= st.slider("Select year for visualisation", min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=(int(df['year'].min()), int(df['year'].max())))
# filtered_data
    filtered_data = df[df['year'].between(*year_range)]
    last_year = year_range[1]
    
# create col for metrics
    mean_life_expectancy = filtered_data['Life Expectancy (IHME)'].mean()
    median_gdp_per_capita= filtered_data['GDP per capita'].median()
    mean_headcount_ratio_upper_mid_income_povline = filtered_data['headcount_ratio_upper_mid_income_povline'].mean()
    count_countries = filtered_data['country'].nunique()

# create 4 key metrics in 4 columns each with a description
    col1, col2, col3, col4= st.columns(4)
    with col1:
        st.metric(label= "Global average life Expectancy", value=f"{mean_life_expectancy:.0f} years")
    with col2:    
        st.metric(label= "Global median GDP per capita", value=f"${median_gdp_per_capita:.0f}")
    
    with col3:
        st.metric(label="Global Poverty Average", value=f"{mean_headcount_ratio_upper_mid_income_povline:.0f}%")

    with col4:
        st.metric(label="Number of Countries", value=count_countries)

#task 5 in tab 1: in terminal conda install -c plotly plotly
#create a scatterplot of the dataframe filtered according to the slider: x=GDP per capita, y = Life Expectancy (IHME) with hover, log, size, color, title, labels
#you might store the code in an extra plots.py file

    scatter_fig = create_scatter_plot(filtered_data,last_year)
    st.plotly_chart(scatter_fig)


#task 6 in tab 1: create a simple model (conda install scikit-learn -y; Randomforest Regressor): 
# features only 3 columns: ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']; target: 'Life Expectancy (IHME)'
#you might store the code in an extra model.py file
#make input fields for inference of the features (according to existing values in the dataset) and use the model to predict the life expectancy for the input values
#additional: show the feature importance as a bar plot
# Load data
df = load_data('global_development_data.csv')  # Replace with your actual data path

# Train the model (only once; comment this after training)
#model = train_model(df)

# Load the model
try:
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("model file not found. please train the model first.")

# Input fields for inference
st.header("Predict Life Expectancy")
st.write("This model uses timestamp, GDP per capita, and poverty rates to predict Life Expectancy.")
st.write("Model Performance")
mse_value = train_model(df)
st.write(f"Mean Square error (MSE): {mse_value:.2f}")  # Display MSE
gdp_per_capita = st.number_input("Enter GDP per capita (dollars):", min_value=0)
headcount_ratio = st.number_input("ENter poverty rate", min_value=0)
year = st.number_input("Year of prediction:", min_value=int(df['year'].min()), max_value=int(df['year'].max()))

# Button to predict
if st.button("Predict"):
    input_data = [gdp_per_capita, headcount_ratio, year]
    prediction = predict_life_expectancy(model, input_data)
    st.write(f"Predicted Life Expectancy (IHME): {prediction[0]:.2f}")

# Show feature importance
st.subheader("Show Feature Importance")
importances = model.feature_importances_
    #st.write("Feature Importances:", importances)  # Check the importances
plot_feature_importance(model, ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year'])


#task 7 in tab 1: create a map plot like the demo in hello streamlit with 3D bars. use chatgpt or similar 
# to create lat and lon values for each country (e.g. capital as reference)
df_merged = load_map_data(df)
st.title("3D Bar Map Plot")
fig = create_3d_bar_plot(df_merged)
st.plotly_chart(fig)

with tabs[1]:  # Assuming this is where you want the map
    st.title("Life Expectancy Map")
    choropleth_fig = create_choropleth_map(df_merged)  # Or use create_marker_map or create_mapbox_map
    st.plotly_chart(choropleth_fig)
