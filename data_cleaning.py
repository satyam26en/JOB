import pandas as pd
import requests
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from io import BytesIO
from zipfile import ZipFile
import os

def clean_data(zip_file_path: str, output_csv_path: str):
    """
    Cleans the job data from a zip file and saves it to a CSV file.

    Args:
        zip_file_path (str): The path to the zip file containing 'jobs.csv'.
        output_csv_path (str): The path to save the cleaned CSV file.
    """
    print(f"Starting data cleaning process. Input: {zip_file_path}, Output: {output_csv_path}")

    # Load data from zip file
    with ZipFile(zip_file_path) as z:
        if 'jobs.csv' not in z.namelist():
            print("Error: 'jobs.csv' not found in the zip file.")
            return
        with z.open('jobs.csv') as f:
            jobs_df = pd.read_csv(f)
    print("Successfully loaded 'jobs.csv' from zip file.")

    # Initial Examination (optional, for debugging or info)
    print(f"Initial dataset shape: {jobs_df.shape}")
    print(f"Missing values before cleaning:\n{jobs_df.isnull().sum()}")

    # Data Cleaning and Preparation
    jobs_df.rename(columns={'resposibilities': 'responsibilities'}, inplace=True)
    print("Renamed 'resposibilities' to 'responsibilities'.")

    # Handling missing values
    cols_to_dropna = ['job_role', 'company', 'posted_on', 'job_link', 'company_link']
    jobs_df.dropna(subset=cols_to_dropna, inplace=True)
    print(f"Dropped rows with NaNs in essential columns: {cols_to_dropna}. Shape after drop: {jobs_df.shape}")

    jobs_df['experience'].fillna('Not specified', inplace=True)
    jobs_df['location'].fillna('Unknown', inplace=True)
    jobs_df['responsibilities'].fillna('Not specified', inplace=True)
    print("Filled NaNs in 'experience', 'location', 'responsibilities'.")

    jobs_df['rating'] = pd.to_numeric(jobs_df['rating'], errors='coerce')
    jobs_df['reviews'] = jobs_df['reviews'].astype(str).str.replace(' Reviews', '', regex=False).str.replace(',', '', regex=False)
    jobs_df['reviews'] = pd.to_numeric(jobs_df['reviews'], errors='coerce')
    print("Cleaned 'rating' and 'reviews' columns.")

    median_rating = jobs_df['rating'].median()
    median_reviews = jobs_df['reviews'].median()
    jobs_df['rating'].fillna(median_rating, inplace=True)
    jobs_df['reviews'].fillna(median_reviews, inplace=True)
    print(f"Filled NaNs in 'rating' with median: {median_rating} and 'reviews' with median: {median_reviews}.")

    # Correcting data types and dropping duplicates
    jobs_df['job_id'] = jobs_df['job_id'].astype(str)
    jobs_df['experience'] = jobs_df['experience'].str.strip()
    jobs_df['salary'] = jobs_df['salary'].str.strip()
    jobs_df['location'] = jobs_df['location'].str.strip()
    jobs_df.drop_duplicates(inplace=True)
    print(f"Corrected data types and dropped duplicates. Shape after deduplication: {jobs_df.shape}")

    # EDA and Feature Engineering

    # job_id cleaning
    jobs_df.drop_duplicates(subset=['job_id'], inplace=True) # Ensure job_id is unique
    print(f"Dropped duplicates based on 'job_id'. Shape: {jobs_df.shape}")

    # job_role cleaning
    jobs_df['job_role'] = jobs_df['job_role'].str.strip().str.title()
    print("Standardized 'job_role'.")

    # company cleaning
    jobs_df['company'] = jobs_df['company'].str.strip().str.title()
    print("Standardized 'company'.")

    # experience processing
    def extract_upper_range(experience):
        match = re.search(r'(\d+)-(\d+)', str(experience))
        if match:
            return int(match.group(2))
        match = re.search(r'(\d+)\+', str(experience))
        if match:
            return int(match.group(1))
        match = re.search(r'(\d+)', str(experience)) # Fallback for single numbers
        if match:
            return int(match.group(1))
        return None

    jobs_df['upper_experience'] = jobs_df['experience'].apply(extract_upper_range)

    def categorize_experience(upper_experience):
        if upper_experience is None:
            return 'Unknown'
        elif upper_experience <= 1:
            return 'Fresher'
        elif upper_experience <= 3:
            return 'Junior'
        elif upper_experience <= 5:
            return 'Medium'
        elif upper_experience <= 10:
            return 'Senior'
        else:
            return 'Expert'

    jobs_df['experience_category'] = jobs_df['upper_experience'].apply(categorize_experience)
    print("Processed 'experience' and created 'experience_category'.")

    # salary processing
    def extract_upper_salary(salary):
        salary_str = str(salary)
        if salary_str.lower() == "not disclosed":
            return None

        # Standardize format: remove 'PA.', 'p.a.', spaces, and commas
        salary_str = salary_str.replace('PA.', '').replace('p.a.', '').replace(',', '').strip()

        # Look for ranges like "XXXXXXX - XXXXXXX"
        match = re.search(r'\d+\s*-\s*(\d+)', salary_str)
        if match:
            return int(match.group(1))

        # Look for single numbers if no range is found
        match = re.search(r'(\d+)', salary_str)
        if match:
            return int(match.group(1))

        return None


    jobs_df['upper_salary'] = jobs_df['salary'].apply(extract_upper_salary)

    def categorize_salary(salary_val):
        if pd.isna(salary_val):
            return 'Not Disclosed'
        elif salary_val < 300000:
            return 'Low'
        elif salary_val < 600000:
            return 'Medium'
        elif salary_val < 1000000:
            return 'Good'
        else:
            return 'High'

    jobs_df['salary_band'] = jobs_df['upper_salary'].apply(categorize_salary)
    print("Processed 'salary' and created 'salary_band', 'upper_salary'.")

    # location processing
    jobs_df['location'] = jobs_df['location'].str.strip().str.title().fillna('Unknown')

    # Filter out common but less specific locations before clustering for better results
    common_locations = ['Permanent Remote', 'Unknown']
    df_for_clustering = jobs_df[~jobs_df['location'].isin(common_locations)].copy()

    if not df_for_clustering.empty and df_for_clustering['location'].nunique() > 1 :
        vectorizer_loc = TfidfVectorizer(stop_words='english', min_df=5) # min_df to avoid rare terms
        try:
            X_loc = vectorizer_loc.fit_transform(df_for_clustering['location'])
            num_clusters_loc = min(10, df_for_clustering['location'].nunique()) # Adjust num_clusters
            if num_clusters_loc > 0:
                kmeans_loc = KMeans(n_clusters=num_clusters_loc, random_state=42, n_init='auto')
                df_for_clustering['location_cluster_temp'] = kmeans_loc.fit_predict(X_loc)

                cluster_replacements = {}
                for cluster_num in range(num_clusters_loc):
                    cluster_locs = df_for_clustering[df_for_clustering['location_cluster_temp'] == cluster_num]['location']
                    if not cluster_locs.empty:
                        most_common_term = cluster_locs.mode()[0] # Use mode for most frequent
                        for loc in cluster_locs.unique():
                            cluster_replacements[loc] = most_common_term

                # Apply replacements to the original DataFrame
                jobs_df['location'] = jobs_df['location'].replace(cluster_replacements)
                print("Processed 'location' using TF-IDF and KMeans clustering.")
            else:
                print("Not enough unique locations to perform clustering after filtering.")
        except ValueError as e:
            print(f"Could not perform location clustering: {e}")
            # This can happen if vocab is empty, e.g. all locations are stop words or too rare
    else:
        print("Not enough diverse data for location clustering or all locations are common/unknown.")


    # reviews (numeric_reviews)
    def extract_review_number(review_str):
        if pd.isna(review_str):
            return None
        # Ensure it's a string, then find numbers
        numbers = re.findall(r'\d+', str(review_str))
        return int(numbers[0]) if numbers else None

    jobs_df['numeric_reviews'] = jobs_df['reviews'].apply(extract_review_number)
    print("Created 'numeric_reviews'.")

    # responsibilities (keyword extraction - simplified from notebook)
    # The notebook's keyword extraction for responsibilities was for EDA, not direct use in the final CSV per its last cell.
    # If specific keywords are needed as columns, this part would need expansion.
    # For now, we'll keep the cleaned 'responsibilities' text.
    jobs_df['responsibilities'] = jobs_df['responsibilities'].str.lower().str.replace('[^a-zA-Z\s]', '', regex=True)
    print("Cleaned 'responsibilities' text (lowercase, removed special chars).")

    # Drop intermediate and unused columns, rename final columns
    df_final = jobs_df.copy()

    # Columns to drop as per notebook's final transformation
    cols_to_drop_final = ['posted_on', 'job_link', 'company_link',
                          'upper_experience', # Original 'experience' is kept, 'experience_category' is the new one
                          # 'salary', # Original 'salary' is kept, 'salary_band' is new
                          'location_cluster' if 'location_cluster' in df_final.columns else None, # if it was created
                          # 'reviews' # Original 'reviews' is kept, 'numeric_reviews' is new
                         ]
    cols_to_drop_final = [col for col in cols_to_drop_final if col and col in df_final.columns] # Filter out None and non-existent
    df_final.drop(columns=cols_to_drop_final, inplace=True, errors='ignore')
    print(f"Dropped intermediate columns: {cols_to_drop_final}")

    # Rename columns as per notebook's final step
    # df_final.rename(columns={'experience_category': 'experience', # This was the notebook logic
    #                          'salary_band': 'salary',
    #                          'numeric_reviews': 'reviews'}, inplace=True)
    # Keeping original experience, salary, reviews and their processed counterparts for flexibility in Streamlit app
    # The Streamlit apps use 'experience_category', 'upper_salary', 'numeric_reviews' etc.
    # So, we will save these valuable columns.

    # The notebook saves a file called 'updated_jobs.csv'. We are saving to 'Clean_Job_File.csv'
    # The columns in 'Clean_Job_File.csv' used by streamlit apps are:
    # 'job_id', 'job_role', 'company', 'experience_category', 'upper_salary', 'location', 'responsibilities'
    # 'rating', 'reviews' (numeric), 'salary_band'

    # Let's select and rename to match the expected Clean_Job_File.csv structure for the apps
    final_columns = {
        'job_id': 'job_id',
        'job_role': 'job_role',
        'company': 'company',
        'experience': 'original_experience', # Keep original experience
        'experience_category': 'experience_category',
        'salary': 'original_salary', # Keep original salary string
        'upper_salary': 'upper_salary', # This is used by result_1.py
        'salary_band': 'salary_band',
        'location': 'location',
        'rating': 'rating',
        'reviews': 'reviews', # This should be the numeric reviews
        'responsibilities': 'responsibilities',
        # 'numeric_reviews' will be the 'reviews' column
    }
    df_to_save = jobs_df.copy()
    df_to_save['reviews'] = df_to_save['numeric_reviews'] # Ensure 'reviews' is numeric

    # Select only the columns needed and rename them if necessary
    df_selected = pd.DataFrame()
    for new_name, old_name in final_columns.items():
        if old_name in df_to_save.columns:
            df_selected[new_name] = df_to_save[old_name]
        elif new_name in df_to_save.columns: # If old_name is same as new_name
             df_selected[new_name] = df_to_save[new_name]
        else:
            print(f"Warning: Column '{old_name}' (for '{new_name}') not found in DataFrame. It will be skipped.")

    # Ensure all expected columns by Streamlit apps are present, fill with NA if not.
    expected_app_columns = ['job_id', 'job_role', 'company', 'experience_category', 'upper_salary', 'location', 'responsibilities', 'rating', 'reviews', 'salary_band']
    for col in expected_app_columns:
        if col not in df_selected.columns:
            df_selected[col] = pd.NA # Or some other default like None or np.nan
            print(f"Warning: Expected column '{col}' was not generated, adding it as NA.")


    print(f"Final DataFrame shape: {df_selected.shape}")
    print(f"Final columns: {df_selected.columns.tolist()}")

    # Save the updated DataFrame
    try:
        df_selected.to_csv(output_csv_path, index=False)
        print(f"Successfully saved cleaned data to {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV to {output_csv_path}: {e}")

if __name__ == '__main__':
    # This part allows the script to be run directly
    # For example, from the command line: python data_cleaning.py
    # Assumes 'jobs.zip' is in the same directory as the script
    # and 'Clean_Job_File.csv' will be saved in the same directory.

    current_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(current_dir, 'jobs.zip')
    csv_output_path = os.path.join(current_dir, 'Clean_Job_File.csv')

    if not os.path.exists(zip_path):
        print(f"Error: Input file 'jobs.zip' not found at {zip_path}")
        # Attempt to download if not found (as per original notebook)
        print("Attempting to download 'jobs.zip'...")
        url = 'https://github.com/satyam26en/JOB/blob/main/jobs.zip?raw=true'
        try:
            response = requests.get(url)
            response.raise_for_status() # Raise an exception for HTTP errors
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            print(f"'jobs.zip' downloaded successfully to {zip_path}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download 'jobs.zip': {e}")
            exit(1) # Exit if download fails

    clean_data(zip_path, csv_output_path)

    # Verify output file
    if os.path.exists(csv_output_path):
        print(f"Verification: '{csv_output_path}' created successfully.")
        df_check = pd.read_csv(csv_output_path)
        print(f"Head of created file:\n{df_check.head()}")
        print(f"Info of created file:\n")
        df_check.info()
    else:
        print(f"Verification failed: '{csv_output_path}' was not created.")
