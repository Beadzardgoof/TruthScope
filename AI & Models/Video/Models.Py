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
import tensorflow_addons as tfa


# Expiremental learning rate scheduler using cosine decay
lr_schedule = tf.keras.experimental.CosineDecayRestarts(
    initial_learning_rate= 0.001,
    first_decay_steps= 1000
)

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
    model.add(Dropout(0.3))

    # Fully connected layers
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    #Compile the model
    model.compile(optimizer=Adam(), 
              loss='binary_crossentropy', 
              metrics=['accuracy', Precision(), Recall(), AUC()])
    
    # model.compile(optimizer=SGD(learning_rate=lr_schedule),
    #            loss='binary_crossentropy', 
    #            metrics=['accuracy', Precision(), Recall(), AUC()])

    return model



# 3D CNN based architecture
def build_3D_CNN():
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv3D(32, kernel_size=(3, 3, 1), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Conv3D(16, kernel_size=(1, 1, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(), 
              loss='binary_crossentropy', 
              metrics=['accuracy', Precision(), Recall(), AUC()])
    
    return model


# Trains the model on the processed data
def train_model(model, output_path, name = "CNN_LSTM" , data_path="Saved Processed Data and Models/[Preprocessed Data]" , batch_size = 8, epochs = 30, use_validation_set = False):
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
    
    # Evaluate the checkpoint best model  on test set
    model_checkpoint = load_model(os.path.join(output_path, 'Checkpoint.h5'))
    evaluation = model_checkpoint.evaluate(X_test, y_test)
    print(f'Checkpoint best model: Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}')

    # Rename it to contain its metrics
    old_path = os.path.join(output_path, 'Checkpoint.h5')
    new_path = os.path.join(output_path, name + ' {:.2f} Lss'.format(evaluation[0]) + ' {:.2f} Acc.h5'.format(evaluation[1] * 100))
    os.rename(old_path, new_path)

    return history, evaluation


