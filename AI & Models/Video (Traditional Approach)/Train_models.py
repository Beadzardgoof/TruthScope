from Preprocess import prepare_dataset
import Models as md
import tensorflow as tf
import os



#with tf.device('/cpu:0'): # For using CPU instead of GPU (CPU has more memory but GPU is faster)


# Real Life Trial Dataset:
    
#path = "Saved Processed Data and Models/court_trial 100x64x64x1"
#prepare_dataset(output_folder= path, dataset= 'court_trial', frame_size=[64, 64], return_gray= True, num_sampled_frames=100)

#cnn = md.build_CNN(input_shape = [100,64,64,1])
#cnn_lstm = md.build_CNN_LSTM(input_shape = [100,64,64,1])
#cnn_3d = md.build_3D_CNN(input_shape = [100,64,64,1])

#md.train_model(data_path = path, output_path = path, model = cnn_3d, name = "CNN_3D", batch_size= 8)


# Miami University Dataset:

#path = "Saved Processed Data and Models/MU3D 100x64x64x1 expiremental 0.7 balanced by downsampling"
#prepare_dataset(output_folder= path, dataset= 'MU3D', frame_size=[64, 64], return_gray= True, num_sampled_frames=100, verify_faces = False, test_size = 20)

#cnn_3d = md.build_3D_CNN_ALT(input_shape = [100,64,64,1])
#cnn_lstm = md.build_CNN_LSTM_ALT(input_shape = [100,64,64,1])

#md.train_model(data_path = path, output_path = path, model = cnn_3d, batch_size = 8, name= "CNN_3D")




