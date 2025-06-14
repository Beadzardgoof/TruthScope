import os
import sys
import numpy as np
import tensorflow as tf
# Force CPU usage to bypass GPU/CuDNN issues
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
from tqdm import tqdm  # Changed from tqdm.notebook to regular tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Get the absolute path to the Video directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Get the AI & Models directory

# Add both the Video directory and its parent to Python path
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now import the local modules
import models as md
import Preprocess as pp

class DeceptionDataset:
    def __init__(self, base_path=None):
        """
        Initialize the dataset handler
        
        Args:
            base_path: Path to your dataset. If None, uses local dataset directory
        """
        if base_path is None:
            base_path = os.path.join(current_dir, 'dataset')
        self.base_path = base_path
        self.truth_path = os.path.join(base_path, 'truth')
        self.deception_path = os.path.join(base_path, 'deception')
        
    def create_mock_dataset(self, num_videos=10, frames_per_video=100, frame_size=(64, 64)):
        """
        Create a mock dataset for testing
        
        Args:
            num_videos: Number of videos to create for each class
            frames_per_video: Number of frames per video
            frame_size: Size of each frame (height, width)
        """
        # Create directories if they don't exist
        os.makedirs(self.truth_path, exist_ok=True)
        os.makedirs(self.deception_path, exist_ok=True)
        
        # Create mock videos
        for i in tqdm(range(num_videos), desc="Creating mock videos"):
            # Truth videos (more consistent patterns)
            truth_frames = np.random.normal(0.5, 0.1, (frames_per_video, *frame_size, 1))
            np.save(os.path.join(self.truth_path, f'truth_{i}.npy'), truth_frames)
            
            # Deception videos (more variable patterns)
            deception_frames = np.random.normal(0.5, 0.2, (frames_per_video, *frame_size, 1))
            np.save(os.path.join(self.deception_path, f'deception_{i}.npy'), deception_frames)
    
    def load_dataset(self, frame_size=(64, 64), num_frames=100):
        """
        Load and preprocess the dataset
        
        Args:
            frame_size: Size to resize frames to
            num_frames: Number of frames to sample per video
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Load truth videos
        truth_files = [f for f in os.listdir(self.truth_path) if f.endswith('.npy')]
        deception_files = [f for f in os.listdir(self.deception_path) if f.endswith('.npy')]
        
        # Combine and shuffle
        all_files = [(os.path.join(self.truth_path, f), 1) for f in truth_files] + \
                   [(os.path.join(self.deception_path, f), 0) for f in deception_files]
        np.random.shuffle(all_files)
        
        # Split into train/test
        split_idx = int(len(all_files) * 0.8)
        train_files = all_files[:split_idx]
        test_files = all_files[split_idx:]
        
        # Load and preprocess videos
        X_train = np.array([np.load(f[0]) for f in train_files])
        y_train = np.array([f[1] for f in train_files])
        X_test = np.array([np.load(f[0]) for f in test_files])
        y_test = np.array([f[1] for f in test_files])
        
        return X_train, X_test, y_train, y_test

def train_models():
    # Initialize dataset
    dataset = DeceptionDataset()
    
    # Create mock dataset if needed
    dataset.create_mock_dataset()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = dataset.load_dataset()
    
    print("Dataset shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Create output directory
    output_path = os.path.join(current_dir, "Saved Processed Data and Models", "local_training")
    os.makedirs(output_path, exist_ok=True)
    
    # Train CNN-LSTM model
    print("\nTraining CNN-LSTM model...")
    cnn_lstm = md.build_CNN_LSTM_ALT(input_shape=[100, 64, 64, 1])
    md.train_model(X_train, X_test, y_train, y_test, 
                  output_path=output_path, 
                  model=cnn_lstm, 
                  name="CNN_LSTM", 
                  batch_size=4, 
                  epochs=30)
    
    # Train 3D CNN model
    print("\nTraining 3D CNN model...")
    cnn_3d = md.build_3D_CNN_ALT()
    md.train_model(X_train, X_test, y_train, y_test, 
                  output_path=output_path, 
                  model=cnn_3d, 
                  name="CNN_3D", 
                  batch_size=4, 
                  epochs=35)
    
    # Plot training results
    plot_training_results(output_path)

def plot_training_results(output_path):
    """Plot training results for both models"""
    plt.figure(figsize=(15, 5))
    
    # Plot CNN-LSTM results
    plt.subplot(1, 2, 1)
    cnn_lstm_history_path = os.path.join(output_path, 'CNN_LSTM_history.npy')
    if os.path.exists(cnn_lstm_history_path):
        cnn_lstm_history = np.load(cnn_lstm_history_path, allow_pickle=True).item()
        plt.plot(cnn_lstm_history['accuracy'], label='Training Accuracy')
        plt.plot(cnn_lstm_history['val_accuracy'], label='Validation Accuracy')
        plt.title('CNN-LSTM Training Results')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No CNN-LSTM history available', 
                horizontalalignment='center', verticalalignment='center')
        plt.title('CNN-LSTM Training Results')
    
    # Plot 3D CNN results
    plt.subplot(1, 2, 2)
    cnn_3d_history_path = os.path.join(output_path, 'CNN_3D_history.npy')
    if os.path.exists(cnn_3d_history_path):
        cnn_3d_history = np.load(cnn_3d_history_path, allow_pickle=True).item()
        plt.plot(cnn_3d_history['accuracy'], label='Training Accuracy')
        plt.plot(cnn_3d_history['val_accuracy'], label='Validation Accuracy')
        plt.title('3D CNN Training Results')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No 3D CNN history available', 
                horizontalalignment='center', verticalalignment='center')
        plt.title('3D CNN Training Results')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'training_results.png'))
    plt.close()

if __name__ == "__main__":
    # Run training
    train_models() 