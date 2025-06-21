import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os # Added for robustness in file path handling, though not strictly necessary if Clean_Job_File.csv is always in root

st.set_page_config(layout="wide")

# --- Data Loading ---
# It's better to load data once and cache it, or ensure data_cleaning.py has run.
# For this refactor, we'll assume Clean_Job_File.csv exists.
# The Streamlit scripts load this CSV. We should centralize this.

DATA_FILE_URL = "Clean_Job_File.csv" # Changed from GitHub URL to local file

@st.cache_data # Cache the data loading
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Error: The data file '{file_path}' was not found. Please run `data_cleaning.py` first.")
        # Optionally, trigger data_cleaning.py or provide instructions
        # For now, we'll exit or return None, and pages should handle this.
        return None
    try:
        df = pd.read_csv(file_path)
        # Basic data integrity checks / cleaning that might be needed if data_cleaning.py wasn't perfect
        df['company'] = df['company'].fillna('').astype(str).str.strip()
        df['experience_category'] = df['experience_category'].fillna('').astype(str).str.strip()
        df['responsibilities'] = df['responsibilities'].fillna('').astype(str)
        df['location'] = df['location'].fillna('').astype(str).str.strip()
        # Ensure numeric columns are numeric, handling potential errors
        df['upper_salary'] = pd.to_numeric(df['upper_salary'], errors='coerce').fillna(0)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce') # Median fill is in data_cleaning
        df['reviews'] = pd.to_numeric(df['reviews'], errors='coerce') # Median fill is in data_cleaning

        return df
    except Exception as e:
        st.error(f"Error loading or processing data from '{file_path}': {e}")
        return None

df_jobs = load_data(DATA_FILE_URL)

# --- Page Functions ---

def page_experience_analysis(df):
    if df is None:
        st.warning("Data not available for Experience Level Analysis.")
        return

    st.title('Unlock Your Career Potential: Experience-Level Salary Insights & Job Openings')

    # Calculate the average upper salary per experience category
    # Ensure 'upper_salary' is numeric. It should be from data_cleaning.py
    average_salary = df.groupby('experience_category')['upper_salary'].mean().reset_index()
    average_salary.columns = ['experience_category', 'average_upper_salary']
    average_salary['average_upper_salary'] = average_salary['average_upper_salary'].round().astype(int)

    # Calculate the number of job openings per experience category
    job_openings = df.groupby('experience_category')['job_id'].count().reset_index()
    job_openings.columns = ['experience_category', 'job_openings']

    category_stats = pd.merge(average_salary, job_openings, on='experience_category')
    category_stats = category_stats.sort_values(by='average_upper_salary')

    fig = px.bar(category_stats,
                 x='experience_category',
                 y='average_upper_salary',
                 title='Average Upper Salary and Job Openings by Experience Category',
                 labels={'experience_category': 'Experience Category', 'average_upper_salary': 'Average Upper Salary (INR)'},
                 text='average_upper_salary',
                 color='experience_category',
                 hover_data=['job_openings'])

    fig.update_layout(
        title={'text': 'Average Salary and Number of Job Openings', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
               'font': {'color': 'black', 'size': 20, 'family': 'Arial, sans-serif'}},
        xaxis={'title': {'text': 'Experience Category', 'font': {'size': 16}},
               'tickfont': {'size': 12}, 'showgrid': False},
        yaxis={'title': {'text': 'Average Salary (INR)', 'font': {'size': 16}},
               'tickfont': {'size': 12}, 'showgrid': True, 'gridcolor': 'lightgray'},
        legend={'font': {'size': 12}},
        uniformtext_minsize=8, uniformtext_mode='hide',
        plot_bgcolor='whitesmoke', paper_bgcolor='whitesmoke'
    )
    st.plotly_chart(fig, use_container_width=True)

def page_location_analysis(df):
    if df is None:
        st.warning("Data not available for Location Analysis.")
        return

    st.title("Job Openings Analysis by Location")
    st.subheader("Visualizing the Top Locations with Maximum Job Openings in India")

    df_loc = df.copy()
    df_loc['location'] = df_loc['location'].fillna('').str.strip()
    # Exclude 'Permanent Remote' and 'Unknown' as per original script for this specific map/chart
    df_loc = df_loc[~df_loc['location'].isin(['Permanent Remote', 'Unknown', ''])]


    top_locations = df_loc['location'].value_counts().reset_index()
    top_locations.columns = ['location', 'Number of Openings']

    # Coordinates might need to be managed more robustly, e.g., from a separate file or geocoding
    location_coordinates = {
        'Bangalore/Bengaluru': [12.9716, 77.5946], 'Hyderabad/Secunderabad': [17.3850, 78.4867],
        'Mumbai': [19.0760, 72.8777], 'Delhi / Ncr': [28.7041, 77.1025], # NCR is broad, using Delhi
        'Gurgaon/Gurugram': [28.4595, 77.0266], 'Pune': [18.5204, 73.8567],
        'Chennai': [13.0827, 80.2707], 'Noida': [28.5355, 77.3910],
        # Add more if data_cleaning produces other major hubs, or make this dynamic
        'Ahmedabad': [23.0225, 72.5714], 'Kolkata': [22.5726, 88.3639]
    }

    # Filter top_locations to only those we have coordinates for
    top_locations_filtered = top_locations[top_locations['location'].isin(location_coordinates.keys())].copy() # Use .copy()

    if top_locations_filtered.empty:
        st.warning("No locations with available coordinates found in the filtered data.")
        return

    top_locations_filtered['lat'] = top_locations_filtered['location'].apply(lambda x: location_coordinates[x][0])
    top_locations_filtered['lon'] = top_locations_filtered['location'].apply(lambda x: location_coordinates[x][1])

    def create_hover_text_loc(location, count):
        return f"Location: {location}<br>Number of Openings: {count}"

    top_locations_filtered['Hover Text'] = top_locations_filtered.apply(
        lambda row: create_hover_text_loc(row['location'], row['Number of Openings']), axis=1
    )

    # Bar Chart
    fig_bar = px.bar(
        top_locations_filtered, x='location', y='Number of Openings',
        title='Top Locations (with coordinates) by Job Openings',
        labels={'location': 'Location', 'Number of Openings': 'Number of Openings'},
        color='location', color_discrete_sequence=px.colors.qualitative.Pastel,
        custom_data=['Hover Text']
    )
    fig_bar.update_layout(
        title={'y':0.9, 'x':0.5, 'xanchor':'center', 'yanchor':'top', 'font': {'size': 20}},
        xaxis_title='Location', yaxis_title='Number of Openings',
        uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False,
        yaxis=dict(tickmode='auto', title='Number of Openings', linecolor='black', linewidth=1, mirror=True),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=40, r=40, b=80, t=80, pad=4),
        xaxis=dict(linecolor='black', linewidth=1, mirror=True)
    )
    fig_bar.update_traces(hovertemplate='%{customdata[0]}')
    st.plotly_chart(fig_bar, use_container_width=True)

    st.write("---") # Separator

    # Map
    mapbox_access_token = "pk.eyJ1Ijoic2F0eWFtMjZlbiIsImEiOiJjbHVkZHN5YWowNTB6MmtvNnN1bWJkYm56In0.z922Z039G7g9grZ2LP6_rQ" # Replace with a valid token if needed or use default
    # Check if mapbox token is set, if not, try OpenStreetMap
    map_style = 'carto-positron'
    if mapbox_access_token and mapbox_access_token.startswith("pk."):
         px.set_mapbox_access_token(mapbox_access_token)
    else: # Fallback or if token is invalid
        st.warning("Mapbox token not properly set. Using default OpenStreetMap style if available, or map may not render correctly.")
        map_style = "open-street-map"


    fig_map = px.scatter_mapbox(
        top_locations_filtered, lat='lat', lon='lon', size='Number of Openings',
        size_max=30, text='location',hover_name='location',
        hover_data={'lat': False, 'lon': False, 'Number of Openings': True},
        color='Number of Openings', color_continuous_scale=px.colors.sequential.Plasma,
        opacity=0.7, title='Geographic Distribution of Job Openings'
    )
    fig_map.update_layout(
        mapbox=dict(center=dict(lat=20.5937, lon=78.9629), zoom=3.8, style=map_style),
        title={'y':0.9, 'x':0.5, 'xanchor':'center', 'yanchor':'top', 'font': {'size': 20}},
        margin=dict(l=40, r=40, b=80, t=80, pad=4)
    )
    st.plotly_chart(fig_map, use_container_width=True)


def page_top_companies(df):
    if df is None:
        st.warning("Data not available for Top Companies Analysis.")
        return

    st.title('Top Hiring Companies by Job Openings')

    experience_levels = ['All'] + df['experience_category'].unique().tolist()
    experience_level = st.selectbox(
        'Select Experience Level:',
        experience_levels,
        index=0
    )

    df_filtered = df.copy()
    if experience_level != 'All':
        df_filtered = df_filtered[df_filtered['experience_category'] == experience_level]

    top_companies = df_filtered['company'].value_counts().reset_index()
    top_companies.columns = ['Company', 'Number of Openings']
    top_companies = top_companies.head(10)

    fig = px.bar(
        top_companies, x='Company', y='Number of Openings',
        title=f'Top 10 Hiring Companies ({experience_level})',
        labels={'Company': 'Company', 'Number of Openings': 'Number of Openings'},
        color='Company', color_discrete_sequence=px.colors.qualitative.Pastel,
        custom_data=['Company']
    )

    # Dynamic Y-axis based on data
    max_openings = top_companies['Number of Openings'].max()
    yaxis_range = [0, max_openings * 1.1] # Add 10% padding

    fig.update_layout(
        title={'y':0.9, 'x':0.5, 'xanchor':'center', 'yanchor':'top', 'font': {'size': 20}},
        xaxis_title='Company', yaxis_title='Number of Openings',
        uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False,
        yaxis=dict(tickmode='auto', range=yaxis_range, title='Number of Openings', linecolor='black', linewidth=1, mirror=True),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=40, r=40, b=80, t=80, pad=4),
        xaxis=dict(linecolor='black', linewidth=1, mirror=True)
    )
    fig.update_traces(hovertemplate='%{customdata[0]}<br>Number of Openings: %{y}')
    st.plotly_chart(fig, use_container_width=True)

def page_job_roles_analysis(df):
    if df is None:
        st.warning("Data not available for Job Roles Analysis.")
        return

    st.title("Job Roles Analysis based on Responsibilities")

    # Ensure responsibilities is not empty and has enough variation
    if df['responsibilities'].nunique() < 2 : # Need at least 2 unique responsibilities for vectorizer
        st.warning("Not enough unique job responsibilities data to perform clustering-based role analysis.")
        return

    vectorizer = TfidfVectorizer(stop_words='english', max_features=500, min_df=5) # Adjusted max_features, added min_df
    try:
        X = vectorizer.fit_transform(df['responsibilities'])
    except ValueError as e:
        st.warning(f"Could not vectorize responsibilities (e.g. all stop words or too few unique terms): {e}")
        return


    num_clusters = min(10, X.shape[0]) # Ensure num_clusters is not more than samples
    if num_clusters < 2: # KMeans needs at least 1 cluster, but practically 2 for meaningful results.
        st.warning(f"Not enough data points ({X.shape[0]}) to form meaningful clusters for job roles.")
        return

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto', max_iter=300) # n_init='auto' is new default

    try:
        df_copy = df.copy() # Work on a copy
        df_copy['Cluster'] = kmeans.fit_predict(X)
    except Exception as e:
        st.error(f"Error during K-Means clustering: {e}")
        return

    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    # Auto-generate cluster names from top 2 terms if possible
    cluster_to_job_role_auto = {}
    for i in range(num_clusters):
        top_terms_list = [terms[ind] for ind in order_centroids[i, :2] if ind < len(terms)]
        cluster_name = ', '.join(top_terms_list).title() if top_terms_list else f"Cluster {i}"
        cluster_to_job_role_auto[i] = cluster_name

    df_copy['Job Role Title'] = df_copy['Cluster'].map(cluster_to_job_role_auto)

    st.subheader("Top Job Categories by Openings (derived from responsibilities)")

    experience_levels_roles = ['All'] + df_copy['experience_category'].unique().tolist()
    experience_type_roles = st.selectbox(
        "Select Experience Level for Role Analysis:",
        experience_levels_roles,
        key='role_experience_selectbox' # Unique key for this selectbox
    )

    df_roles_filtered = df_copy.copy()
    if experience_type_roles != 'All':
        df_roles_filtered = df_roles_filtered[df_roles_filtered['experience_category'] == experience_type_roles]

    openings_by_role_cluster = df_roles_filtered.groupby('Job Role Title')['job_id'].count().reset_index()
    openings_by_role_cluster.columns = ['Job Role Category', 'Number of Openings']
    openings_by_role_cluster = openings_by_role_cluster.sort_values(by='Number of Openings', ascending=False).head(10) # Top 10

    fig_roles = px.bar(
        openings_by_role_cluster,
        x='Job Role Category', y='Number of Openings',
        title=f'Top Job Categories ({experience_type_roles} Level)',
        labels={'Job Role Category': 'Job Role Category (from Responsibilities)', 'Number of Openings': 'Number of Openings'},
        color='Job Role Category', color_discrete_sequence=px.colors.qualitative.Safe
    )

    max_role_openings = openings_by_role_cluster['Number of Openings'].max()
    role_yaxis_range = [0, max_role_openings * 1.1]

    fig_roles.update_layout(
        xaxis_title='Job Role Category', yaxis_title='Number of Openings',
        title={'y':0.9, 'x':0.5, 'xanchor':'center', 'yanchor':'top', 'font': {'size': 18}},
        uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False,
        yaxis=dict(tickmode='auto', range=role_yaxis_range, title='Number of Openings', linecolor='black', linewidth=1, mirror=True),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=40, r=40, b=150, t=80, pad=4), # Increased bottom margin for longer role titles
        xaxis=dict(linecolor='black', linewidth=1, mirror=True, tickangle=-45) # Angled ticks for readability
    )
    st.plotly_chart(fig_roles, use_container_width=True)

    with st.expander("See Top Terms per Job Role Category"):
        for i in range(min(num_clusters, 10)): # Show details for up to 10 clusters
            role_name = cluster_to_job_role_auto.get(i, f"Cluster {i}")
            st.write(f"**{role_name}**: `{[terms[ind] for ind in order_centroids[i, :5] if ind < len(terms)]}`")


# --- Main App Structure ---
PAGES = {
    "Experience Level Salary & Openings": page_experience_analysis,
    "Job Openings by Location": page_location_analysis,
    "Top Hiring Companies": page_top_companies,
    "Job Roles Analysis (from Responsibilities)": page_job_roles_analysis
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Call the selected page function
page_function = PAGES[selection]

if df_jobs is not None:
    page_function(df_jobs)
else:
    st.error("Dataset could not be loaded. Please ensure `Clean_Job_File.csv` exists in the root directory and `data_cleaning.py` has been run successfully.")

st.sidebar.info("This app analyzes job posting data. Ensure `data_cleaning.py` has been run to generate/update `Clean_Job_File.csv`.")
