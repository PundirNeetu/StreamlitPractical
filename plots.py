import plotly.express as px

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