import os
import numpy as np
import matplotlib.pyplot as plt

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "Saved Processed Data and Models", "local_training")

def plot_training_results():
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
    plt.show()  # This will display the plot

if __name__ == "__main__":
    plot_training_results() 