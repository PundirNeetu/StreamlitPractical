import plotly.express as px
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go


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
    df = pd.read_csv('global_development_data.csv')

# Load the world shapefile from GeoPandas
    world = gpd.read_file(r"C:\Users\neetu\Downloads\DSR\DSRNotesGit\week6\Streamlit18Oct\StreamlitPractical\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp")
# Merge your DataFrame with the world DataFrame based on country names
    df_merged = world.merge(df, left_on='NAME', right_on='country', how='inner')
    df_merged = df_merged.to_crs(epsg=3857)  # Using Web Mercator for example

# Now you can get the centroid (latitude, longitude) of each country
    df_merged['latitude'] = df_merged['geometry'].centroid.y
    df_merged['longitude'] = df_merged['geometry'].centroid.x
    return df_merged

def create_3d_bar_plot(df):
    
    fig = go.Figure()

    # Add bars for each country
    for index, row in df.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row['longitude']],
            y=[row['latitude']],
            z=[0],  # Start the bars at z=0
            mode='lines+markers',
            line=dict(width=5),  # Width of the bars
            marker=dict(size=row['Life Expectancy (IHME)']*10, color='blue'),  # Size based on height
            name=row['country'],  # Use country name for labeling
        ))
    fig.add_trace(go.Scatter3d(
            x=[row['longitude'], row['longitude']],
            y=[row['latitude'], row['latitude']],
            z=[0, row['Life Expectancy (IHME)']],  # Height of the bar
            mode='lines',
            line=dict(width=5, color='blue'),
        ))

    fig.update_layout(
        title='3D Bar Plot for Each Country',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Value',  # Replace with a relevant title
        ),
        showlegend=True
    )

    return fig


