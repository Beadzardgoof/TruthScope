from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, SimpleRNN, GRU, Embedding, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras_nlp.layers import SinePositionEncoding
import os
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from joblib import dump, load
import seaborn as sns


### Deep learning approaches ###

# Models architecture 
def build_lstm(input_shape, output_units=2, learning_rate=0.0001):
    # Model architecture
    model = Sequential()
    model.add(LSTM(units=32, return_sequences=True, input_shape=input_shape, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(LSTM(units=32,  return_sequences=False, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(units=output_units, activation='softmax'))
    
    # Compile the model with a specified learning rate
    adam_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam_optimizer,
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def build_lstm_bidirectional(input_shape, output_units=2, learning_rate=0.0001):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=32, activation='tanh'), input_shape=input_shape))
    model.add(Dense(units=output_units, activation='softmax'))
    
    adam_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def build_simple_rnn(input_shape, output_units=2, learning_rate=0.0001):
    model = Sequential()
    model.add(SimpleRNN(units=32, activation='tanh', input_shape=input_shape))
    model.add(Dense(units=output_units, activation='softmax'))
    
    adam_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def build_gru(input_shape, output_units=2, learning_rate=0.0001):
    model = Sequential()
    model.add(GRU(units=32, activation='tanh', input_shape=input_shape))
    model.add(Dense(units=output_units, activation='softmax'))
    
    adam_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def build_ann(output_units=1, learning_rate=0.01):
    model = Sequential()
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dense(units=output_units, activation='sigmoid'))
    
    adam_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam_optimizer,
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model



# Trains the model on the processed data
def train_model_dl(model, output_path, X_train, y_train, X_test, y_test, name = "model" , batch_size = 8, epochs = 30):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Define checkpoint
    checkpoint = ModelCheckpoint(
        os.path.join(output_path, 'Checkpoint.h5') , save_weights_only=False, save_best_only=True, verbose=1
    )
    
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


    # Train the model
    history = model.fit(X_train, y_train, epochs= epochs, validation_data=(X_test, y_test), callbacks=[checkpoint],  batch_size=batch_size)
    
    # Evaluate the checkpoint best model  on test set
    model_checkpoint = load_model(os.path.join(output_path, 'Checkpoint.h5'))
    #model_checkpoint = model
    evaluation = model_checkpoint.evaluate(X_test, y_test)
    print(evaluation)
    print(f'Checkpoint best model: Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}')

    # Rename it to contain its metrics
    old_path = os.path.join(output_path, 'Checkpoint.h5')
    new_path = os.path.join(output_path, name + ' {:.2f} Lss'.format(evaluation[0]) + ' {:.2f} Acc.h5'.format(evaluation[1] * 100))
    os.rename(old_path, new_path)

    return history, evaluation
    
  


### Machine learning approaches ###

# Draws confusion matrix as well as important metrics like accuracy, recall etc.
def calculate_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test, name):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true_test, y_pred_test)

    # Calculate other metrics
    accuracy_train = accuracy_score(y_true_train, y_pred_train)
    accuracy_test = accuracy_score(y_true_test, y_pred_test)
    recall = recall_score(y_true_test, y_pred_test, average='weighted')
    precision = precision_score(y_true_test, y_pred_test, average='weighted')
    f1 = f1_score(y_true_test, y_pred_test, average='weighted')

    # Print the metrics
    print(f"### {name} ###")
    print(f"Train Accuracy: {accuracy_train:.4f}")
    print(f"Test Accuracy: {accuracy_test:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test F1 Score: {f1:.4f}\n")

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()
    return [accuracy_test, recall, precision, f1]


def train_model_ml(model,output_path, X_train, y_train, X_test, y_test, name = "model"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    metrics = calculate_metrics(y_train, y_pred_train, y_test, y_pred_test, name)
    dump(model, os.path.join(output_path, name + ' {:.2f} Acc.joblib'.format(metrics[0] * 100)))
    return metrics

