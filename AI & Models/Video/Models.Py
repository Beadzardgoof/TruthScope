import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, LSTM, BatchNormalization, TimeDistributed, Conv2D, MaxPooling2D, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Defines the architecture for a CNN-LSTM model
def build_CNN_LSTM(input_shape):
    model = Sequential()

    # TimeDistributed wrapper to apply CNN across time dimension
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))

    # LSTM layer for temporal processing
    model.add(LSTM(50, return_sequences=False))

    # Fully connected layers
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

    return model


# Trains the model on the processed data
def train_model(model, name = "CNN_LSTM" , data_path="Saved Processed Data and Models/[Preprocessed Data]", batch_size = 8, epochs = 30):
    # Load saved preprocessed arrays for train, test, and eval datasets
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'Y_train.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'Y_test.npy'))

    # Define checkpoint
    checkpoint = ModelCheckpoint(
        os.path.join(data_path, 'Checkpoint.h5') , save_weights_only=False, save_best_only=True, verbose=1
    )
    
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)


    # Train the model
    history = model.fit(X_train, y_train, epochs= epochs, validation_data=(X_test, y_test), callbacks=[checkpoint],  batch_size=batch_size)

    # Evaluate the model on the test set
    evaluation = model.evaluate(X_test, y_test)
    print(f'Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}')

    # Save trained model with accuracy up to two decimal places
    model.save(os.path.join(data_path, name + ' {:.2f} Acc.h5'.format(evaluation[1] * 100)))

    return history, evaluation

