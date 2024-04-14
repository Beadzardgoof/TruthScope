import Preprocess as pp
import Models as md
import tensorflow as tf
import os
import gc
#with tf.device('/cpu:0'): # For using CPU instead of GPU (CPU has more memory but GPU is faster)

### Real Life Trial Dataset ###

# Preprocess the video data
output_path = "Saved Processed Data and Models/court_trial 100x64x64x1 MTCNN"
#pp.prepare_dataset(output_folder= output_path, dataset= 'court_trial', frame_size=[64, 64], return_gray= True, num_frames_to_sample=100)

    
# Get data and manually split
data_path = "Saved Processed Data and Models/court_trial 100x64x64x1 MTCNN/Numpy Arrays"
manual_split_video_names_path = '../Datasets/Real Life Trial Cases Data/Manual Split Test Videos.txt'
X_train, X_test, y_train, y_test = pp.get_manual_split_data(data_path, manual_split_video_names_path)

# Printing shapes
print('X_train shape is:' , X_train.shape)
print('X_test shape is:' , X_test.shape)

# Train the models

#1 CNN-LSTM
cnn_lstm = md.build_CNN_LSTM_ALT(input_shape = [100,64,64,1])
md.train_model(X_train, X_test, y_train, y_test, output_path = output_path, model = cnn_lstm, name = "CNN_LSTM", batch_size= 4, epochs = 30)

#2 3D CNN
cnn_3d = md.build_3D_CNN_ALT()
md.train_model(X_train, X_test, y_train, y_test , output_path = output_path, model = cnn_3d, name = "CNN_3D", batch_size= 4, epochs = 35)



### Miami University Dataset ###

