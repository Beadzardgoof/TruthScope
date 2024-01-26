import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



# Helper function that gets all video pathes and labels, default path is to the Real Life Trial Cases Dataset.
def get_videos_paths(dataset = "court_trial"):
    if dataset == 'court_trial':
        base_path = "../Datasets/Real Life Trial Cases Data/Clips"
    else: base_path = "../Datasets/MU3D-Package/Videos"
    
    categories = ["Deceptive", "Truthful"]
    videos = []

    for category in categories:
        category_path = os.path.join(base_path, category)
        for video_file in os.listdir(category_path):
            video_path = os.path.join(category_path, video_file)
            label = 1 if category == "Deceptive" else 0
            videos.append((video_path, label))
    return videos
    

# Preprocesses a single video
##1 Crop the face from each frame video
##2 Sample video into same number of frames
##3 Resize all frames to a similar size
##4 Apply Eulerian magnification algorithm on the frames
def preprocess_video(video_path, eulerian_magnify, num_sampled_frames, frame_size, return_gray, verify_faces):
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
    while len(verified_faces) < num_sampled_frames:
        verified_faces += verified_faces[:max(1, num_sampled_frames - len(verified_faces))]

    # Sample the verified faces
    frame_sample_rate = max(1, len(verified_faces) // num_sampled_frames)
    processed_frames = [verified_faces[i] for i in range(0, len(verified_faces), frame_sample_rate)]

    # Limit the number of frames to the desired number
    processed_frames = processed_frames[:num_sampled_frames]

    # Process each frame
    for i in range(len(processed_frames)):
        face = processed_frames[i]

        # Convert to grayscale if necessary
        if return_gray:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Apply Eulerian magnification if necessary
        if eulerian_magnify:
            face = apply_eulerian_magnification(face)

        processed_frames[i] = face

    # Normalize the frames
    processed_frames = np.array(processed_frames) / 255.0
    channels = 1 if return_gray else 3
    processed_frames = processed_frames.reshape((num_sampled_frames, frame_size[0], frame_size[1], channels))

    return processed_frames

# Eulerian mangification algorithm by MIT, it magnifies colors of videos to reveal tiny changes that may hold valuable information.
def apply_eulerian_magnification(frame):
    # Implement the Eulerian magnification algorithm here
    # This part is left as a placeholder for the actual implementation
    pass

# Helper function that Takes processed frames and saves them as images in output path (for visualization)
def save_frames_as_images(processed_frames, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save each frame as an image
    for i, frame in enumerate(processed_frames):
        # Denormalize to show real color
        frame = frame * 255.0
        # Define the filename for each image
        filename = os.path.join(output_folder, f'frame_{i:04d}.png')  # Saves as frame_0001.png, frame_0002.png, etc.
        # Write the image
        cv2.imwrite(filename, frame)

# Helper function to get last two folders in a path string (to)
def get_last_two_folders_concatenated(path):
    # Split the path into components
    parts = path.split(os.sep)
    
    # Extract the last two components
    last_two = parts[-2:] if len(parts) >= 2 else parts

    # Remove the file extension from the last component if it's a file
    last_two[-1] = os.path.splitext(last_two[-1])[0]

    # Concatenate the last two components with a '/'
    return '/'.join(last_two)

# Main function to prepare the dataset, it preprocesses every video in the dataset and saves train, eval and test sets as .npy files as well as images for visualization
def prepare_dataset(output_folder, dataset="court_trial", eulerian_magnify=False, num_sampled_frames=100, frame_size=[64, 64], return_gray=True, verify_faces=True, test_size=0.2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all video paths and corresponding labels
    videos = get_videos_paths(dataset)
    video_paths, labels = zip(*videos)

    # Split data into train, test, eval
    train_paths, test_paths, train_labels, test_labels = train_test_split(video_paths, labels, test_size=test_size)

    datasets = {
        "train": (train_paths, train_labels),
        "test": (test_paths, test_labels),
    }

    # Process and save each dataset
    for name, (paths, labels) in datasets.items():
        data = []
        for video_path, label in zip(paths, labels):
            processed_frames = preprocess_video(video_path, eulerian_magnify, num_sampled_frames, frame_size, return_gray, verify_faces)
            print(video_path, "\n")
            save_frames_as_images(processed_frames, os.path.join(output_folder, name, get_last_two_folders_concatenated(video_path))) # Ex: OutputFolder/Train/Deceptive/trial_lie_0021
            data.append((processed_frames, label))
        
        X, y = zip(*data)
        X = np.array(X)
        y = np.array(y)

        # Save preprocessed data
        np.save(os.path.join(output_folder, f'X_{name}.npy'), X)
        np.save(os.path.join(output_folder, f'Y_{name}.npy'), y)