import plotly.express as px
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
from math import pi


def create_scatter_plot(df, last_year): 
    #create a scatterplot of the dataframe filtered according to the slider: x=GDP per capita, y = Life Expectancy (IHME) with hover, 
    # log, size = head_count_ration_upper_mid_income_povline, 
    # color=country, title, labels
    # Create a scatter plot using Plotly Express
    fig = px.scatter(
        data_frame=df,
        x='GDP per capita',
        log_x= True,
        y='Life Expectancy (IHME)',
        color='country',
        size='headcount_ratio_upper_mid_income_povline',
        #hover_data=[x_column, y_column, color_column, size_column]
    )
    
    # Update the layout
    fig.update_layout(
        title_text=f'Scatter Plot of Life Expectancy vs GDP per capita ({last_year})',
        xaxis_title='GDP per Capita (log Scale)',
        yaxis_title='Life Expectancy (years)',
        )
    
    return fig


def load_map_data(df):


# Load your dataset
    df = pd.read_csv('./data/global_development_data.csv')

# Load the world shapefile from GeoPandas
    world = gpd.read_file("./data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
# Merge your DataFrame with the world DataFrame based on country names
    df_merged = world.merge(df, left_on='NAME', right_on='country', how='inner')
    df_merged = df_merged.to_crs(epsg=3857)  # Using Web Mercator for example

# Now you can get the centroid (latitude, longitude) of each country
    df_merged['latitude'] = df_merged['geometry'].centroid.y
    df_merged['longitude'] = df_merged['geometry'].centroid.x
    return df_merged





