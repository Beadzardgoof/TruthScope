import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import gc
import imageio
import random
from moviepy.editor import VideoFileClip
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import librosa
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
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
    

# Helper functions to apply stft noise reduction
def threshold_stft(stft, threshold=1e-5):
    # Apply threshold to the magnitude of the STFT
    magnitude = np.abs(stft)
    stft_thresholded = np.where(magnitude < threshold, 0, stft)
    return stft_thresholded

def stft_noise_reduction(audio_array):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio_array)
    
    # Threshold the STFT
    stft_thresholded = threshold_stft(stft)
    
    # Compute the inverse STFT
    audio_array = librosa.istft(stft_thresholded, length=len(audio_array))  # Ensure the length matches
    return audio_array

def scale_and_apply_pca(x, num_pca_components=5, is_test = False):
    if is_test:
        scaler = load('StandardScaler.joblib')
        pca = load('PCA.joblib')
        x = scaler.transform(x)
        x = pca.transform(x)
        pass
    else:
        scaler = StandardScaler()
        pca = PCA(n_components= num_pca_components)
        x = scaler.fit_transform(x)
        x = pca.fit_transform(x)
        dump(pca, 'PCA.joblib')
        dump(scaler, 'StandardScaler.joblib')
    return x


# Gets audio from a video, removes background noise and finally extracts MFCC features (frame_length=0.025, hop_length=0.01, num_samples= 700)
def get_audio_features(video_path, target_sr=44100, frame_length=0.025, hop_length=0.01, num_samples= 700):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    # Extract audio to numpy array
    audio_array = audio.to_soundarray(fps=target_sr, nbytes=2)
    
    # Ensure it's mono
    if audio_array.shape[1] > 1:
        audio_array = np.mean(audio_array, axis=1)

    # Apply STFT-based noise reduction
    audio_array = stft_noise_reduction(audio_array)

    # Compute the MFCC features
    mfcc_features = librosa.feature.mfcc(y=audio_array, sr=target_sr, n_mfcc=13, n_fft=int(target_sr * frame_length), hop_length=int(target_sr * hop_length))
    
    # Compute mean of MFCC features across all frames
    # mean_mfcc = np.mean(mfcc_features, axis=1)
    # return mean_mfcc

    N = mfcc_features.shape[1]
    if N < num_samples:
        # If N is less than num_samples, repeat samples in a circular manner
        indices = np.tile(np.arange(N), int(np.ceil(num_samples/N)))[:num_samples]
    else:
        # Calculate step size for regular sampling
        step_size = N // num_samples
        # Generate indices for regular sampling
        indices = np.arange(0, N, step_size)[:num_samples]
    
    sampled_mfcc = mfcc_features[:, indices]
    
    return sampled_mfcc

# Helper function to get last folder/file in a path string
def get_video_name_from_path(path):
    # Split the path into components
    parts = path.split(os.sep)
    
    # Extract the last component
    last = parts[-1] if len(parts) >= 1 else parts

    # Remove the file extension from the last component if it's a file
    last = os.path.splitext(last)[0]

    return last


# Main function to prepare the dataset, it preprocesses every video in the dataset and saves train and test sets as .npy files as well as images for visualization
def prepare_dataset(output_folder, dataset="court_trial"):
    
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
        processed_frames = get_audio_features(video_path)
        
        label_text = "Deceptive" if label == 1 else "Truthful"
        
        # Save as numpy array
        video_name = get_video_name_from_path(video_path)            
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

  
