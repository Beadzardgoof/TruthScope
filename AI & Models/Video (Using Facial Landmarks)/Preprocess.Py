import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import gc
import dlib
from sklearn.preprocessing import MinMaxScaler
from keras_preprocessing.sequence import pad_sequences

# Helper function that gets all video pathes and labels, default path is to the Real Life Trial Cases Dataset.
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
    
    
def get_videos_paths_mu3d(use_truthprop = False, truthprop_threshold = 0.5):
    base_path = "../Datasets/MU3D-Package/Videos"
    
    videos = []
    
    # Load the cookbook
    cookbook_path = "../Datasets/MU3D-Package/MU3D Codebook.xlsx"
    df = pd.read_excel(cookbook_path, sheet_name='Video-Level Data')

    for video_file in os.listdir(base_path):
            video_path = os.path.join(base_path, video_file)
            # Split the base name and extension (e.g., ('WM027_4PL', '.wmv'))
            name, _ = os.path.splitext(video_file)
            
            # Use thresholded truth prop as label or just use provided labels in Veracity column
            if(use_truthprop):
                truth_prop = df.loc[df['VideoID'] == name, 'TruthProp'].values[0]
                label = 0 if truth_prop >= truthprop_threshold else 1
            else:
                label = df.loc[df['VideoID'] == name, 'Veracity'].values[0]
                
            videos.append((video_path, label))
    return videos
    
    
# Preprocesses a single video
def preprocess_video(video_path, sampling_time=0.05, sequence_length=10):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    
    # Initialize face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    # Frame rate and interval calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * sampling_time)
    
    # Store landmarks for each frame
    landmarks_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if len(faces) > 0:
                landmarks = predictor(gray, faces[0])
                landmark_coords = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
                landmarks_list.append(np.array(landmark_coords).flatten())
    cap.release()

    if len(landmarks_list) == 0:
        return np.empty((0, 136))
    
    # Normalize landmarks
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_landmarks = scaler.fit_transform(landmarks_list)
    
    # Create sequences
    sequences = [normalized_landmarks[i:i + sequence_length] for i in range(len(normalized_landmarks) - sequence_length + 1)]
    print(np.array(sequences).shape)
    return np.array(sequences)
   


# Main function to prepare the dataset, it preprocesses every video in the dataset and saves train and test sets as .npy files as well as images for visualization
def prepare_dataset(output_folder, dataset="court_trial", test_size=0.2, sequence_length= 10):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all video paths and corresponding labels
    if dataset == "court_trial":
        videos = get_videos_paths_court_trial()
    else:
        videos = get_videos_paths_mu3d()    
    
    
    video_paths, labels = zip(*videos)

    # Split data into train and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(video_paths, labels, test_size=test_size, random_state=42, stratify=labels)

    datasets = {
        "train": (train_paths, train_labels),
        "test": (test_paths, test_labels),
    }

    # Loop over train and test sets
    for name, (paths, labels) in datasets.items():
        print(f"Preprocessing {name} data")
        
        all_sequences = []
        all_labels = []
        
        # Loop over videos
        for video_path, label in zip(paths, labels):
            print(video_path, "\n")
            sequences = preprocess_video(video_path, sequence_length=sequence_length)
            if sequences.shape[0] > 0:
                all_sequences.append(sequences)
                all_labels.extend([label] * sequences.shape[0])
                
        # Pad sequences and fix shapes
        sequences_padded = np.vstack(all_sequences)
        labels_array = np.array(all_labels)
        
        print(sequences_padded.shape)
        print(labels_array.shape)
        
        # Save combined data
        np.save(os.path.join(output_folder, f'X_{name}.npy'), sequences_padded)
        np.save(os.path.join(output_folder, f'Y_{name}.npy'), labels_array)
        
        # Free up memory
        del sequences_padded, labels_array, all_sequences, all_labels
        gc.collect()
        
        
        
  
  
  
  
