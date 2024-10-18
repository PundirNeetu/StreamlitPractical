import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


    
