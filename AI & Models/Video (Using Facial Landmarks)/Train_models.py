from Preprocess import prepare_dataset
import Models as md
import tensorflow as tf
import os



#with tf.device('/cpu:0'): # For using CPU instead of GPU (CPU has more memory but GPU is faster)


# Real Life Trial Dataset:
    
#path = "Saved Processed Data and Models/court_trial"
#prepare_dataset(output_folder= path, dataset= 'court_trial')

#cnn = md.build_CNN(input_shape = [100,64,64,1])
#cnn_lstm = md.build_CNN_LSTM(input_shape = [100,64,64,1])
#cnn_3d = md.build_3D_CNN(input_shape = [100,64,64,1])

#md.train_model(data_path = path, output_path = path, model = cnn, name = "CNN", batch_size= 8)


# Miami University Dataset:

path = "Saved Processed Data and Models/MU3D"
#prepare_dataset(output_folder= path, dataset= 'MU3D')
lstm = md.build_LSTM(input_shape = [10, 136])
md.train_model(data_path = path, output_path = path, model = lstm, batch_size = 8, name= "LSTM")




