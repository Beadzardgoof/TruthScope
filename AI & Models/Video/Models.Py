import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, LSTM, BatchNormalization, TimeDistributed, Conv2D, MaxPooling2D, Bidirectional
from keras.optimizers import Adam, SGD, Adamax, Adadelta, Ftrl, Nadam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.applications import ResNet50
from keras.optimizers import RMSprop
from keras.metrics import Precision, Recall, AUC
#import tensorflow_addons as tfa


# Defines the architecture for a CNN-LSTM model
def build_CNN_LSTM(input_shape):
    model = Sequential()

    # TimeDistributed wrapper to apply CNN across time dimension
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))

    # LSTM layer for temporal processing
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.3))

    # Fully connected layers
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    #Compile the model
    model.compile(optimizer=Adam(), 
              loss='binary_crossentropy', 
              metrics=['accuracy', AUC()])
    

    return model


# 3D CNN based architecture
def build_3D_CNN():
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.3))

    
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    #Compile the model
    model.compile(optimizer=Adam(), 
              loss='binary_crossentropy', 
              metrics=['accuracy', AUC()])

    
    return model

# LSTM architecture
def build_LSTM(input_shape):
    model = Sequential()
    
   # LSTM layer for temporal processing
    model.add(LSTM(32, return_sequences=True, input_shape = input_shape, activation = 'tanh'))
    model.add(LSTM(32, return_sequences=False, input_shape = input_shape, activation = 'tanh'))

    # Fully connected layers
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    #Compile the model
    model.compile(optimizer=Adam(), 
              loss='binary_crossentropy', 
              metrics=['accuracy', AUC()])

    
    return model

# Trains the model on the processed data
def train_model(X_train, X_test, y_train, y_test, model, output_path, name = "CNN_LSTM", batch_size = 8, epochs = 30):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Define checkpoint
    checkpoint = ModelCheckpoint(
        os.path.join(output_path, 'Checkpoint.h5') , save_weights_only=False, save_best_only=True, verbose=1,   monitor='val_accuracy', mode='max',  
    )
    
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)


    # Train the model
    history = model.fit(X_train, y_train, epochs= epochs, validation_data=(X_test, y_test), callbacks=[checkpoint],  batch_size=batch_size)
    
    # Evaluate the checkpoint best model  on test set
    model_checkpoint = load_model(os.path.join(output_path, 'Checkpoint.h5'))
    evaluation = model_checkpoint.evaluate(X_test, y_test)
    print(f'Checkpoint best model: Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}')

    # Rename it to contain its metrics
    old_path = os.path.join(output_path, 'Checkpoint.h5')
    new_path = os.path.join(output_path, name + ' {:.2f} Lss'.format(evaluation[0]) + ' {:.2f} Acc.h5'.format(evaluation[1] * 100))
    os.rename(old_path, new_path)

    return history, evaluation


