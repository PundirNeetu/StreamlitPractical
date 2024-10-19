import plotly.express as px

def create_choropleth_map(df):
    fig = px.choropleth(
        df,
        locations='country',
        locationmode='country names',
        color='Life Expectancy (IHME)',
        hover_name='country',
        color_continuous_scale=px.colors.sequential.Plasma,
        labels={'Life Expectancy (IHME)': 'Life Expectancy'},
        title='Life Expectancy by Country'
    )
    
    fig.update_geos(projection_type="mercator")
    fig.update_layout(height=600, width=800) 
    return fig
def create_marker_map(df):
    fig = px.scatter_geo(
        df,
        locations='country',
        locationmode='country names',
        size='Life Expectancy (IHME)',
        hover_name='country',
        title='Life Expectancy by Country',
        projection='natural earth',
        size_max=40
    )
    fig.update_layout(height=600, width=800) 
    return fig
import plotly.express as px

def create_mapbox_map(df):
    fig = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        size='Life Expectancy (IHME)',
        hover_name='country',
        mapbox_style='carto-positron',
        size_max=50,
        title='Life Expectancy by Country'
    )
    fig.update_layout(mapbox=dict(center=dict(lat=0, lon=0), zoom=1))
    fig.update_layout(height=600, width=800) 
    return fig
