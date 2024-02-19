import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, LSTM, BatchNormalization, TimeDistributed, Conv2D, MaxPooling2D, Bidirectional, GRU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.optimizers import RMSprop



# Defines the architecture for an LSTM
def build_LSTM(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

    return model

# Defines the architecture for GRU
def build_GRU(input_shape):
    model = Sequential()
    model.add(GRU(64, input_shape=(input_shape[1], input_shape[2]), return_sequences=True))
    model.add(GRU(64))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

    return model


# Trains the model on the processed data
def train_model(model, output_path, name = "LSTM" , data_path="Saved Processed Data and Models/[Preprocessed Data]" , batch_size = 8, epochs = 30, use_validation_set = False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Load saved preprocessed arrays for train and test sets
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'Y_train.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'Y_test.npy'))
    
        
    
    # Further split test data into test and validation sets.
    if(use_validation_set):
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
        validation_data = (X_val, y_val)
    else:
        validation_data = (X_test, y_test)

    
    # Define checkpoint
    checkpoint = ModelCheckpoint(
        os.path.join(output_path, 'Checkpoint.h5') , save_weights_only=False, save_best_only=True, verbose=1
    )
    
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)


    # Train the model
    history = model.fit(X_train, y_train, epochs= epochs, validation_data=validation_data, callbacks=[checkpoint],  batch_size=batch_size)

    # Evaluate the model on the test set
    evaluation = model.evaluate(X_test, y_test)
    print(f'Last model: Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}')

    # Evaluate the checkpoint best model too on test set
    model_checkpoint = load_model(os.path.join(output_path, 'Checkpoint.h5'))
    evaluation = model_checkpoint.evaluate(X_test, y_test)
    print(f'Chekpoint best model: Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}')

    # Save trained model with accuracy up to two decimal places
    model.save(os.path.join(output_path, name + ' {:.2f} Acc.h5'.format(evaluation[1] * 100)))

    return history, evaluation


