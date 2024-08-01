# -*- coding: utf-8 -*-
"""Result_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17aZrPDSdJ3utmJ_-oy59Yl_nHIdc2stQ
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# Streamlit app title
st.title('Experience Category-wise Average Salary and Number of Job Openings')

# Load the dataset from the provided URL
file_url = "https://raw.githubusercontent.com/satyam26en/JOB/main/Clean_Job_File.csv"
df = pd.read_csv(file_url)

# Calculate the average upper salary per experience category and round it to the nearest integer
average_salary = df.groupby('experience_category')['upper_salary'].mean().reset_index()
average_salary.columns = ['experience_category', 'average_upper_salary']
average_salary['average_upper_salary'] = average_salary['average_upper_salary'].round().astype(int)

# Calculate the number of job openings per experience category
job_openings = df.groupby('experience_category')['job_id'].count().reset_index()
job_openings.columns = ['experience_category', 'job_openings']

# Merge the two DataFrames
category_stats = pd.merge(average_salary, job_openings, on='experience_category')

# Sort the DataFrame by average upper salary in ascending order
category_stats = category_stats.sort_values(by='average_upper_salary')

# Create a bar chart with different colors for each bar
fig = px.bar(category_stats,
             x='experience_category',
             y='average_upper_salary',
             title='Average Upper Salary and Job Openings by Experience Category',
             labels={'experience_category': 'Experience Category', 'average_upper_salary': 'Average Upper Salary'},
             text='average_upper_salary',
             color='experience_category',  # Assign different colors based on experience category
             hover_data=['job_openings'])

# Update layout for better visualization and background color change
fig.update_layout(
    title={
        'text': 'Average Salary and Number of Job Openings ',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'color': 'black',  # Color for the title
            'size': 24,
            'family': 'Arial, sans-serif'
        }
    },
    xaxis={
        'title': {
            'text': 'Experience Category',
            'font': {
                'color': 'black',  # Color for x-axis title
                'size': 24,
                'family': 'Arial, sans-serif'
            },
        },
        'tickfont': {
            'color': 'black',  # Color for x-axis ticks
            'size': 14,
            'family': 'Arial, sans-serif'
        },
        'showgrid': False  # Optionally, hide x-axis grid lines
    },
    yaxis={
        'title': {
            'text': 'Average Salary',
            'font': {
                'color': 'black',  # Color for y-axis title
                'size': 24,
                'family': 'Arial, sans-serif'
            },
        },
        'tickfont': {
            'color': 'black',  # Color for y-axis ticks
            'size': 14,
            'family': 'Arial, sans-serif'
        },
        'showgrid': True,
        'gridcolor': 'lightgray'  # Use light gray grid lines
    },
    legend={
        'font': {
            'color': 'black',
            'size': 14,
            'family': 'Arial, sans-serif'
        }
    },
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    plot_bgcolor='whitesmoke',  # Change plot background color to whitesmoke
    paper_bgcolor='whitesmoke'  # Change paper background color to whitesmoke
)

# Display the interactive bar chart in Streamlit
st.plotly_chart(fig)