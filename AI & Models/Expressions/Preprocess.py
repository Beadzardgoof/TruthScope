from feat import Detector
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


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
    


# Extracts py-feat features from a given video
def extract_facial_features(video_path, output_path = 'example.csv'):
    detector = Detector(verbose=False)  # Initialize the detector
    fex = detector.detect_video(video_path, skip_frames = 30, batch_size=4)  # Detect facial features in the video once each second
    
    # Save the extracted facial expressions to a CSV file
    fex.to_csv(output_path, index=False)
    
    return fex


# Experiment with this version later
def extract_facial_features_optimized(video_path, frame_skip=30):
    # Initialize the Py-Feat detector
    detector = Detector()  
    
    # Open the video file for processing
    cap = cv2.VideoCapture(video_path)
    
    # Lists to store detection results for each processed frame
    all_landmarks = []  # To store landmarks
    all_aus = []  # To store action units
    all_emotions = []  # To store emotions
    
    frame_count = 0  # Counter to keep track of the current frame
    
    # Loop through each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame
        if not ret:  # If no frame is read (end of video), exit loop
            break
        
        # Only process every 'frame_skip' number of frames to reduce computation
        if frame_count % frame_skip == 0:
            # Detect faces to get bounding boxes
            detected_faces = detector.detect_faces(frame)
            # Detect facial landmarks within the detected faces
            landmarks = detector.detect_landmarks(frame, detected_faces)
            # Detect action units based on the detected landmarks
            aus = detector.detect_aus(frame, landmarks)
            # Detect emotions based on the detected faces and landmarks
            emotions = detector.detect_emotions(frame, detected_faces, landmarks)
            
            # Store the results
            all_landmarks.append(landmarks)
            all_aus.append(aus)
            all_emotions.append(emotions)
        
        frame_count += 1  # Increment the frame counter
    
    # Release the video capture object
    cap.release()
    
    # Return the detected landmarks, action units, and emotions
    return all_landmarks, all_aus, all_emotions
 
        




# Helper function to get last folder/file in a path string
def get_video_name_from_path(path):
    # Split the path into components
    parts = path.split(os.sep)
    
    # Extract the last component
    last = parts[-1] if len(parts) >= 1 else parts

    # Remove the file extension from the last component if it's a file
    last = os.path.splitext(last)[0]

    return last


# Main function to extract features from all videos in the dataset
def extract_features_from_dataset(output_path = "Facial Features Court Trial", dataset="court_trial"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, 'Deceptive'))
        os.makedirs(os.path.join(output_path, 'Truthful'))
    
    # Get all video paths and corresponding labels
    if dataset == "court_trial":
        videos = get_videos_paths_court_trial()
    else:
        videos = get_videos_paths_mu3d()
    
    # Loop over videos
    counter = 0
    for video_path, label in videos:
        counter += 1
        print(counter, video_path, "\n")
        label_text = "Deceptive" if label == 1 else "Truthful"
        try:
            video_name = get_video_name_from_path(video_path)
            fex = extract_facial_features(video_path, os.path.join(output_path, label_text, f'{video_name}.csv'))
            print(f"Sample shape is {fex.shape}")
        except:
            print("Failed, skipping this video")
        

# Loads features csv, selects certain columns and computes the mean across records to get one feature vector
def prepare_feature_vector(features_csv_path):
    # Load the CSV file
    data = pd.read_csv(features_csv_path)
    
    # Include emotions columns
    potential_columns = ['Pitch', 'Roll', 'Yaw', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    columns_to_keep = [col for col in potential_columns if col in data.columns]
    
    # Include AU (action units) columns
    au_columns = [col for col in data.columns if col.startswith('AU')]
    columns_to_keep += au_columns
    
    # Filter the dataframe based on the existing columns
    filtered_data = data[columns_to_keep]
    
    # Compute and return the mean of the records as a numpy array
    feature_vector_mean = filtered_data.mean().to_numpy()
    
    return feature_vector_mean


# Main function to extract features from all videos in the dataset
def prepare_feature_vectors_from_dataset(data_path = "Facial Features Court Trial", output_path = "Feature Vectors Court Trial"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, 'Deceptive'))
        os.makedirs(os.path.join(output_path, 'Truthful'))
    
    
    folder_names = ['truthful', 'deceptive']

    # Loop over the folders in the data path
    for i in range(2): 
        folder_path = os.path.join(data_path, folder_names[i])
        
        # Loop over the files in the folder
        for file_name in os.listdir(folder_path):       
            video_path = os.path.join(folder_path, file_name)
        
            video_name = get_video_name_from_path(video_path)
            
            # Process and save npy arrays
            feature_vector = prepare_feature_vector(video_path)
            np.save(os.path.join(output_path, folder_names[i], f'{video_name}.npy'), feature_vector)
            
            print(f"Sample shape is {feature_vector.shape}")
            
            
# Function that reads saved numpy arrays from a path and makes a train test split based on manually picked test video ids (for court trial data)
def get_manual_split_data(data_path, test_set_videos_names_path):
    # Read the list of test video names from the .txt file
    with open(test_set_videos_names_path, 'r') as file:
        test_videos = file.read().splitlines()
    
    # Initialize lists to hold the data and labels
    X_train, X_test, y_train, y_test = [], [], [], []
    
    # Mapping for the labels
    labels_map = {'truthful': 0, 'deceptive': 1}
    
    # Loop over the folders in the data path
    for folder_name in ['truthful', 'deceptive']:
        folder_path = os.path.join(data_path, folder_name)

        # Loop over the files in the folder
        for file_name in os.listdir(folder_path):       
            video_name = file_name.rsplit('.', 1)[0]  # Remove the .npy extension
            video_path = os.path.join(folder_path, file_name)
            video_data = np.load(video_path)
            
            # Determine if the video is part of the test set
            if video_name in test_videos:
                X_test.append(video_data)
                y_test.append(labels_map[folder_name])
            else:
                X_train.append(video_data)
                y_train.append(labels_map[folder_name])
    
    # Convert lists to numpy arrays before returning
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Shuffle the training and testing data separately
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    
    return X_train, X_test, y_train, y_test
    
   
# Function that reads saved numpy arrays from a path without a split (to do auto train-test split after with MU3D)
def get_data_from_saved_numpy_arrays(data_path):
    # Initialize lists to hold the data and labels
    X, y = [], []
    
    # Mapping for the labels
    labels_map = {'truthful': 0, 'deceptive': 1}
    
    # Loop over the folders in the data path
    for folder_name in ['truthful', 'deceptive']:
        folder_path = os.path.join(data_path, folder_name)

        # Loop over the files in the folder
        for file_name in os.listdir(folder_path):       
            video_path = os.path.join(folder_path, file_name)
            video_data = np.load(video_path)
            
            X.append(video_data)
            y.append(labels_map[folder_name])

    
    # Convert lists to numpy arrays before returning
    X, y = np.array(X), np.array(y)


    # Shuffle the training and testing data separately
    X, y= shuffle(X, y, random_state=42)
    
    return X, y