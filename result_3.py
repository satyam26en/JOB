# -*- coding: utf-8 -*-
"""Untitled55.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1s72atFXuiOYbW2_Ct6wUU-qUMq2s9sQg
"""

import pandas as pd
import plotly.express as px
import streamlit as st


file_url = "https://raw.githubusercontent.com/satyam26en/JOB/main/Clean_Job_File.csv"
df = pd.read_csv(file_url)
df['location'] = df['location'].fillna('').str.strip()
df = df[~df['location'].isin(['Permanent Remote', 'Unknown'])]

# Get the top locations based on the number of job openings
top_locations = df['location'].value_counts().reset_index()
top_locations.columns = ['location', 'Number of Openings']

# Define the coordinates for the top locations
location_coordinates = {
    'Bangalore/Bengaluru': [12.9716, 77.5946],
    'Hyderabad/Secunderabad': [17.3850, 78.4867],
    'Mumbai': [19.0760, 72.8777],
    'Delhi / Ncr': [28.7041, 77.1025],
    'Gurgaon/Gurugram': [28.4595, 77.0266],
    'Pune': [18.5204, 73.8567],
    'Chennai': [13.0827, 80.2707],
    'Noida': [28.5355, 77.3910]
}

# Filter to include only the defined locations
top_locations = top_locations[top_locations['location'].isin(location_coordinates.keys())]

# Add coordinates to the top locations
top_locations['lat'] = top_locations['location'].apply(lambda x: location_coordinates[x][0])
top_locations['lon'] = top_locations['location'].apply(lambda x: location_coordinates[x][1])

# Create detailed hover text for each bar
def create_hover_text(location, count):
    return f"Location: {location}<br>Number of Openings: {count}"

top_locations['Hover Text'] = top_locations.apply(lambda row: create_hover_text(row['location'], row['Number of Openings']), axis=1)

# Create the bar chart using Plotly
fig_bar = px.bar(
    top_locations,
    x='location',
    y='Number of Openings',
    title='Top Locations with Maximum Job Openings in India',
    labels={'location': 'Location', 'Number of Openings': 'Number of Openings'},
    color='location',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    custom_data=['Hover Text']
)

fig_bar.update_layout(
    title={
        'text': 'Top Locations with Maximum Job Openings in India',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title='Location',
    yaxis_title='Number of Openings',
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    showlegend=False,
    yaxis=dict(
        tickmode='array',
        tickvals=[0, 2000, 4000, 6000, 8000, 10000, 12000],
        range=[0, 12000],
        title='Number of Openings',
        linecolor='black', linewidth=2, mirror=True
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=50, r=50, b=100, t=100, pad=4),
    xaxis=dict(linecolor='black', linewidth=2, mirror=True),
    width=1000,
    height=600
)

fig_bar.update_traces(hovertemplate='%{customdata[0]}')

# Display the bar chart using Streamlit
st.plotly_chart(fig_bar)

# Add spacing
st.write(" " * 20)

# Mapbox access token
mapbox_access_token = "pk.eyJ1IjoibWFwbG9vcGVyIiwiYSI6ImNpa3l6cWR4bTAwOXV0em55aDRqOHY4ajMifQ.NPNAJWiTdh1XYLw-SB8YWQ"

# Create the map using Plotly
fig_map = px.scatter_mapbox(
    top_locations,
    lat='lat',
    lon='lon',
    size='Number of Openings',
    size_max=50,
    text='location',
    title='Top Locations with Maximum Job Openings in India',
    hover_name='location',
    hover_data={'lat': False, 'lon': False, 'Number of Openings': True},
    color='Number of Openings',
    color_continuous_scale=px.colors.sequential.Plasma,
    opacity=0.5
)

fig_map.update_layout(
    mapbox=dict(
        accesstoken=mapbox_access_token,
        center=dict(lat=20.5937, lon=78.9629),
        zoom=4,
        style='carto-positron'
    ),
    title={
        'text': 'Top Locations with Maximum Job Openings in India',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    margin=dict(l=50, r=50, b=100, t=100, pad=4),
    width=1000,
    height=600
)

# Display the map using Streamlit
st.plotly_chart(fig_map)
