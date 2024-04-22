import Preprocess as pp
from keras.models import load_model
import numpy as np
import os
import tensorflow as tf

# Set the environment variable for reproducability
os.environ['TF_DETERMINISTIC_OPS'] = '1' 

# Path to the saved model
base_path = 'Saved Processed Data and Models/court_trial 100x64x64x1 MTCNN'
model_path = os.path.join(base_path, 'CNN_3D 0.61 Lss 87.50 Acc [BEST].h5') # Best model


# Load the saved model
model = load_model(model_path)

print("For best results, your sample should be 20-30 seconds long.")
while True:
    print("Enter sample path (or type 'exit' to quit):")
    sample_path = input()
    
    if not os.path.exists(sample_path):
        print("Please Make sure the path is correct.")
        continue
    
    # Break the loop if the user types 'exit'
    if sample_path.lower() == 'exit':
        break

    # Preprocess the input video
    processed_frames = pp.preprocess_video(sample_path, verify_faces = False)
    pp.save_frames_as_gif(processed_frames, 'visualize frames')
    processed_frames = np.expand_dims(processed_frames, axis=0)
    print(processed_frames.shape) 
    
    # Predict
    predictions = model.predict(processed_frames)
    
    # Convert the probability to percentage
    percentage = predictions[0][0] * 100  # Adjust indexing based on model's output shape
    print(f"Prediction: {percentage:.2f}% Liar")





# Generalization test:

#Get data and manually split
# data_path = "Saved Processed Data and Models/court_trial 100x64x64x1 MTCNN/Numpy Arrays"
# manual_split_video_names_path = '../Datasets/Real Life Trial Cases Data/Manual Split Test Videos.txt'

# X_train, X_test, y_train, y_test = pp.get_manual_split_data(data_path, manual_split_video_names_path)

# print(X_train.shape)

# # Evaluate on loaded data
# with tf.device('/cpu:0'): # For using CPU instead of GPU (CPU has more memory but GPU is faster)
#     evaluation = model.evaluate(X_test, y_test)
#     print(f'Evaluation: Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}')
