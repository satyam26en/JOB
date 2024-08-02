# -*- coding: utf-8 -*-
"""Result_2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dwWLqGxynAzpqtaym8CB3le3_IE9R8dg
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load the dataset from the provided GitHub URL
file_url = "https://raw.githubusercontent.com/satyam26en/JOB/main/Clean_Job_File.csv"
df = pd.read_csv(file_url)

# Preprocess the responsibilities text
df['responsibilities'] = df['responsibilities'].fillna('')

# Vectorize the responsibilities text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['responsibilities'])

# Use K-Means clustering to cluster the responsibilities
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42, max_iter=2500, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# Get the top terms per cluster
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

# Function to get the top terms for each cluster
def get_top_terms(cluster_num, n_terms=10):
    top_terms = [terms[ind] for ind in order_centroids[cluster_num, :n_terms]]
    return top_terms

# Display the top terms for each cluster to manually assign job role names
for i in range(num_clusters):
    print(f"Cluster {i} top terms: {get_top_terms(i)}")

# Manually assign job role names based on top terms
cluster_to_job_role = {
    0: 'Software Developer',
    1: 'Sales Executive',
    2: 'HR Manager',
    3: 'Operations Manager',
    4: 'Marketing Specialist',
    5: 'Customer Service',
    6: 'Project Manager',
    7: 'Customer Service',
    8: 'Sales Executive',
    9: 'Business Analyst'
}

df['Job Role'] = df['Cluster'].map(cluster_to_job_role)

# Provided job openings data
job_openings_data = {
    'Software Developer': {'Total': 37944, 'Fresher': 1253, 'Junior': 4013, 'Mid-Level': 8332, 'Senior': 19075, 'Expert': 5371},
    'Project Manager': {'Total': 6528, 'Fresher': 185, 'Junior': 926, 'Mid-Level': 1790, 'Senior': 2838, 'Expert': 789},
    'Data Scientist': {'Total': 2882, 'Fresher': 0, 'Junior': 1, 'Mid-Level': 7, 'Senior': 2852, 'Expert': 22},
    'Finance Manager': {'Total': 11491, 'Fresher': 245, 'Junior': 1085, 'Mid-Level': 2088, 'Senior': 5762, 'Expert': 2311},
    'Business Analyst': {'Total': 3275, 'Fresher': 346, 'Junior': 770, 'Mid-Level': 1477, 'Senior': 486, 'Expert': 196},
    'Sales Executive': {'Total': 2916, 'Fresher': 332, 'Junior': 720, 'Mid-Level': 777, 'Senior': 806, 'Expert': 281},
    'Customer Service': {'Total': 2743, 'Fresher': 75, 'Junior': 366, 'Mid-Level': 615, 'Senior': 1174, 'Expert': 513},
    'HR Manager': {'Total': 2685, 'Fresher': 165, 'Junior': 528, 'Mid-Level': 698, 'Senior': 1028, 'Expert': 266},
    'Marketing Specialist': {'Total': 741, 'Fresher': 5, 'Junior': 7, 'Mid-Level': 15, 'Senior': 691, 'Expert': 23},
    'Operations Manager': {'Total': 1803, 'Fresher': 0, 'Junior': 0, 'Mid-Level': 0, 'Senior': 1800, 'Expert': 3}
}

# Create detailed hover text for each bar
def create_hover_text(job_role, exp_type):
    data = job_openings_data[job_role]
    if exp_type == 'Total':
        return f"Job Role: {job_role}<br>Total Openings: {data['Total']}"
    else:
        return f"Job Role: {job_role}<br>{exp_type} Openings: {data[exp_type]}"

# Streamlit app
st.title('Top Job Openings Based on Responsibilities and Experience Category')

# Add a selectbox for experience type
exp_type = st.selectbox(
    'Select Experience Type:',
    ('Total', 'Fresher', 'Junior', 'Mid-Level', 'Senior', 'Expert')
)

# Create dataframe based on selected experience type
if exp_type == 'Total':
    total_openings_by_role = pd.DataFrame([
        {'Job Role': role, 'Number of Openings': data['Total'], 'Hover Text': create_hover_text(role, exp_type)}
        for role, data in job_openings_data.items()
    ])
else:
    total_openings_by_role = pd.DataFrame([
        {'Job Role': role, 'Number of Openings': data[exp_type], 'Hover Text': create_hover_text(role, exp_type)}
        for role, data in job_openings_data.items()
    ])

total_openings_by_role = total_openings_by_role.sort_values(by='Number of Openings')

# Create a bar chart using Plotly
fig = px.bar(
    total_openings_by_role,
    x='Job Role',
    y='Number of Openings',
    title=f'Top Job Openings Based on Responsibilities and Experience Category ({exp_type})',
    labels={'Job Role': 'Job Role', 'Number of Openings': 'Number of Openings'},
    color='Job Role',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    custom_data=['Hover Text']
)

# Update layout for better visualization and border
fig.update_layout(
    xaxis_title='Job Role',
    yaxis_title='Number of Openings',
    title={
        'text': f'Top Job Openings Based on Responsibilities and Experience Category ({exp_type})',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    showlegend=False,  # Hide legend
    yaxis=dict(
        tickmode='linear',
        dtick=3000,
        range=[0, total_openings_by_role['Number of Openings'].max() + 3000],
        title='Number of Openings',
        linecolor='black', linewidth=2, mirror=True
    ),
    plot_bgcolor='white',  # Set background color
    paper_bgcolor='white',  # Set paper background color
    margin=dict(l=50, r=50, b=100, t=100, pad=4),  # Adjust margins
    xaxis=dict(linecolor='black', linewidth=2, mirror=True),  # Add border to x-axis
    width=1000,  # Width of the chart
    height=600  # Height of the chart
)

# Update hover template to show detailed information
fig.update_traces(hovertemplate='%{customdata[0]}<br>Number of Openings: %{y}')

# Show the interactive bar chart
st.plotly_chart(fig)