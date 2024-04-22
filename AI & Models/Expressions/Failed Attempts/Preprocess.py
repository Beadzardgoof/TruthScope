import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import os
import Signals_Extraction as se

# Helper function that gets all video pathes and labels
def get_videos_paths_court_trial():
    base_path = "../Datasets/Real Life Trial Cases Data/Clips"
    
    categories = ["Deceptive", "Truthful"]
    videos = []

    for category in categories:
        category_path = os.path.join(base_path, category)
        for video_file in os.listdir(category_path):
            video_path = os.path.join(category_path, video_file)
            label = 1 if category == "Deceptive" else 0
            videos.append((video_path, label))
    return videos
       
def get_videos_paths_mu3d():
    base_path = "../Datasets/MU3D-Package/Videos"
    
    videos = []
    
    # Load the cookbook
    cookbook_path = "../Datasets/MU3D-Package/MU3D Codebook.xlsx"
    df = pd.read_excel(cookbook_path, sheet_name='Video-Level Data')

    for video_file in os.listdir(base_path):
            video_path = os.path.join(base_path, video_file)
            # Split the base name and extension (e.g., ('WM027_4PL', '.wmv'))
            name, _ = os.path.splitext(video_file)
            # Get label
            label = df.loc[df['VideoID'] == name, 'Veracity'].values[0]
            videos.append((video_path, label))
    return videos


# Helper function to get last folder/file in a path string
def get_video_name_from_path(path):
    # Split the path into components
    parts = path.split(os.sep)
    
    # Extract the last component
    last = parts[-1] if len(parts) >= 1 else parts

    # Remove the file extension from the last component if it's a file
    last = os.path.splitext(last)[0]

    # Concatenate the last two components with a '/'
    return last

# Load data from a CSV file into a pandas DataFrame with specified column names.
def load_data_from_csv(filename):
    column_names = ['Id', 'avg_bpms', 'blinking', 'hand', 'gaze', 'lips', 'Mood']
    data = pd.read_csv(filename, header=None, names=column_names)
    return data


# Process the dataframe to extract and compute necessary features for prediction,
# including handling missing values and calculating statistics like minimum and maximum heart rates.
# Optionally includes a 'Veracity' column if a truth value is provided.
def extract_features(df, label_data = True):
    # Extract video ID
    df['Video_ID'] = df['Id'].apply(get_video_name_from_path).str.split('.').str[0]

    # Prepare a dataframe for all video IDs
    all_video_ids = pd.DataFrame({'Video_ID': df['Video_ID'].unique()})

    # Calculate min and max heart rates, excluding zeroes
    heart_rates = df[df['avg_bpms'] > 57]
    min_heart_rates = heart_rates.groupby('Video_ID', group_keys=False)['avg_bpms'].min().rename('min_avg_bpms')
    max_heart_rates = heart_rates.groupby('Video_ID', group_keys=False)['avg_bpms'].max().rename('max_avg_bpms')
    
    # Total blinks calculation
    blinks = df.groupby(['Video_ID', 'blinking'], group_keys=False)['blinking'].count().reset_index(name='blink_count').fillna(0)
    total_blinks = blinks[blinks['blinking'] == 2].groupby('Video_ID', group_keys=False)['blink_count'].sum().rename('total_blinks') / 30  # Assuming 30 fps for blink rate

    # Face touches calculation
    df['hand_change'] = (df['hand'] - df['hand'].shift(1)).fillna(0)
    face_touches = df[(df['hand_change'] == 1) & (df['hand'] == 1)].groupby('Video_ID', group_keys=False).size().rename('total_face_touches')

    # Lip compressions detection
    lip_compressions = (df['lips'] < 0.32).astype(int).groupby(df['Video_ID'], group_keys=False).sum().rename('total_lip_compressions')

    # Mood changes calculation, explicitly preventing group keys from being added to the index
    mood_changes = df.groupby('Video_ID', group_keys=False)['Mood'].apply(lambda x: x.ne(x.shift()).cumsum()).reset_index(drop=True)
    mood_change_counts = mood_changes.groupby(df['Video_ID'], group_keys=False).nunique().rename('total_mood_changes')

    # Mood distribution across categories
    mood_distribution = df.groupby(['Video_ID', 'Mood'], group_keys=False).size().unstack(fill_value=0).apply(lambda x: x / x.sum(), axis=1)

    # Compile all features into a single DataFrame
    features_df = all_video_ids.merge(min_heart_rates, how='left', on='Video_ID').merge(max_heart_rates, how='left', on='Video_ID').merge(total_blinks, how='left', on='Video_ID').merge(face_touches, how='left', on='Video_ID').merge(lip_compressions, how='left', on='Video_ID').merge(mood_distribution, how='left', on='Video_ID').merge(mood_change_counts, how='left', on='Video_ID')

    # Calculate the difference between max and min heart rates and add it as a column
    features_df['heart_rate_diff'] = features_df['max_avg_bpms'] - features_df['min_avg_bpms']
    
    # Fill missing values
    features_df.fillna({'max_avg_bpms': 70, 'min_avg_bpms': 60, 'total_face_touches': 0, 'total_blinks': 0, 'heart_rate_diff': 10}, inplace=True)
    
    # Label data (for training only)
    if (label_data): features_df['label'] = features_df['Video_ID'].apply(lambda x: 1 if 'lie' in x else 0)

    return features_df


# Main function to prepare the dataset, it preprocesses every video in the dataset and saves train and test sets as .npy files as well as images for visualization
def prepare_dataset(dataset="court_trial"):
    # Get all video paths and corresponding labels
    if dataset == "court_trial":
        videos = get_videos_paths_court_trial()
    else:
        videos = get_videos_paths_mu3d()
    # Loop over videos
    counter = 0
    for video_path, _ in videos:
        counter += 1
        print(counter, video_path, "\n")
        se.main(video_path, f"signals_{dataset}.csv")
        

# Reads a text file containing video names and returns a list of these names.
def read_test_video_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        video_names = file.read().splitlines()
    return video_names

# Splits the DataFrame into training and test sets based on the video names.
def split_df_based_on_test_names(df, test_video_names):
    # Filter the DataFrame to create the test set
    test_df = df[df['Video_ID'].isin(test_video_names)]
    
    # Filter the DataFrame to create the training set
    train_df = df[~df['Video_ID'].isin(test_video_names)]
    
    return train_df, test_df