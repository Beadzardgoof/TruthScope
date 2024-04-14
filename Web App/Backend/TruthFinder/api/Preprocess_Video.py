import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import gc
import imageio
from sklearn.utils import shuffle
import random
import dlib
from mtcnn import MTCNN
import sys
import mediapipe as mp


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
    
    
# Preprocesses a single video
##1 Crop the face from each frame in the video
##2 Sample video into a given number of frames
##3 Resize all frames to a given size
# (Note MTCNN is accurate and more computationly expensive, so with that specific approach step 1 and 2 or interchanged in order, so sampling is done before cropping the face)

# Uses haar cascades to extract faces, achievies different results for same input (but our best model is trained on it so far unfortunately)
def preprocess_video(video_path, num_frames_to_sample = 100, frame_size = [64,64], return_gray= True, verify_faces = True):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # List to store cropped faces
    cropped_faces = []
    
    # Process the video
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        # Break the loop if there are no more frames
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Crop and store the first face found (if any)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_frame = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_frame, (frame_size[0], frame_size[1]))
            cropped_faces.append(resized_face)

    # Release the video capture object
    cap.release()

    # Verify the faces if required
    if verify_faces:
        verified_faces = [face for face in cropped_faces if len(face_cascade.detectMultiScale(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), 1.1, 4)) > 0]
    else:
        verified_faces = cropped_faces

    # Augment frames if necessary
    while len(verified_faces) < num_frames_to_sample:
        verified_faces += verified_faces[:max(1, num_frames_to_sample - len(verified_faces))]

    # Sample the verified faces
    frame_sample_rate = max(1, len(verified_faces) // num_frames_to_sample)
    processed_frames = [verified_faces[i] for i in range(0, len(verified_faces), frame_sample_rate)]

    # Limit the number of frames to the desired number
    processed_frames = processed_frames[:num_frames_to_sample]

    # Process each frame
    for i in range(len(processed_frames)):
        face = processed_frames[i]

        # Convert to grayscale if necessary
        if return_gray:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        processed_frames[i] = face

    # Normalize the frames
    processed_frames = np.array(processed_frames) / 255.0
    channels = 1 if return_gray else 3
    processed_frames = processed_frames.reshape((num_frames_to_sample, frame_size[0], frame_size[1], channels))


    return processed_frames

# Uses MTCNNs for face extraction (best quality so far)
def preprocess_video_mtcnn(video_path, num_frames_to_sample=100, frame_size=[64, 64], return_gray=True):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Calculate the total number of frames and the sampling rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_sample_rate = max(1, total_frames // num_frames_to_sample)
    
    # Initialize the MTCNN detector
    detector = MTCNN()
    
    # List to store cropped faces
    cropped_faces = []
    
    current_frame = 0
    sampled_frame_count = 0
    
    # Process the video
    while cap.isOpened() and sampled_frame_count < num_frames_to_sample:
        # Set the video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        # Read a frame from the video
        ret, frame = cap.read()
        
        # Break the loop if there are no more frames
        if not ret:
            break
        
        # Detect faces in the frame
        faces = detector.detect_faces(frame)
        
        # Crop and store the first face found (if any)
        if len(faces) > 0:
            x, y, width, height = faces[0]['box']
            face_frame = frame[y:y+height, x:x+width]
            resized_face = cv2.resize(face_frame, (frame_size[0], frame_size[1]))
            cropped_faces.append(resized_face)
            sampled_frame_count += 1
        
        current_frame += frame_sample_rate
    
    # Release the video capture object
    cap.release()
    
    # Augment frames if necessary
    while len(cropped_faces) < num_frames_to_sample:
        cropped_faces += cropped_faces[:max(1, num_frames_to_sample - len(cropped_faces))]
    
    # Ensure the list is not longer than needed
    cropped_faces = cropped_faces[:num_frames_to_sample]
    
    # Process each frame
    for i in range(len(cropped_faces)):
        face = cropped_faces[i]
        
        # Convert to grayscale if necessary
        if return_gray:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        cropped_faces[i] = face
    
    # Normalize the frames
    processed_frames = np.array(cropped_faces) / 255.0
    channels = 1 if return_gray else 3
    processed_frames = processed_frames.reshape((num_frames_to_sample, frame_size[0], frame_size[1], channels))
    
    return processed_frames


# Helper function that Takes processed frames and saves them as gifs in output path (for visualization)
def save_frames_as_gif(processed_frames, output_folder, filename='video.gif', fps=10):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the full path for the gif file
    gif_path = os.path.join(output_folder, filename)

    # Initialize list to hold processed frames for GIF
    gif_frames = []

    # Iterate over the frames and convert each to uint8
    for i in range(processed_frames.shape[0]):
        # Assuming processed_frames is 4D (num_frames, height, width, channels)
        frame = processed_frames[i]
        if frame.ndim == 3 and frame.shape[2] == 1:
            # If frames have a channel dimension, remove it for grayscale
            frame = frame.reshape((frame.shape[0], frame.shape[1]))
        # Normalize to 0-255 and convert to uint8
        frame = (frame * 255).astype(np.uint8)
        gif_frames.append(frame)

    # Save the frames as a gif
    imageio.mimsave(gif_path, gif_frames, fps=fps)
    
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

# Main function to prepare the dataset, it preprocesses every video in the dataset and saves train and test sets as .npy files as well as images for visualization
def prepare_dataset(output_folder, dataset="court_trial", num_frames_to_sample=100, frame_size=[64, 64], return_gray=True, verify_faces=True):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    numpy_arrays_path = os.path.join(output_folder, 'Numpy Arrays')
    if not os.path.exists(numpy_arrays_path):
        os.makedirs(numpy_arrays_path)
        os.makedirs(os.path.join(numpy_arrays_path, 'Deceptive'))
        os.makedirs(os.path.join(numpy_arrays_path, 'Truthful'))

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
        #processed_frames = preprocess_video(video_path, num_frames_to_sample, frame_size)
        processed_frames = preprocess_video_mtcnn(video_path, num_frames_to_sample, frame_size, return_gray)
        
        label_text = "Deceptive" if label == 1 else "Truthful"
        
        # Visualize data and save as numpy array
        video_name = get_video_name_from_path(video_path)
        save_frames_as_gif(processed_frames, os.path.join(output_folder, label_text , video_name))
            
        x = np.array(processed_frames)
        np.save(os.path.join(numpy_arrays_path, label_text, f'{video_name}.npy'), x)
        
        print(f"Sample shape is {x.shape}")


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

    # Shuffle the training and testing data separately (no shuffling in multimodal to maintain consistency among modalities)
    # X_train, y_train = shuffle(X_train, y_train, random_state=42)
    # X_test, y_test = shuffle(X_test, y_test, random_state=42)
    
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


