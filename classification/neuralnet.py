import pandas as pd
import tensorflow as tf

import librosa as lib

import librosa.display

import IPython.display as ipd

import matplotlib.pyplot as plt
import numpy as np
import random as r

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import models
from keras import layers
from keras import regularizers

#This is the individual Neural Network model implemented for the project

def NNModel():
    #Read Pre-processed CSV file
    data = pd.read_csv('data.csv')

    #Drop any unnecessary columns
    data = data.drop(['filename'], axis=1)

    #Extract list of classes
    genre_list = data.iloc[:, -1]
    y = genre_list

    #Normalize all parameters
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

    #Split data into training, and validation sets (90-10 split)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    #Implement model architecture (1 input layer, 5 hidden layers, 1 Output softmax layer)
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(6, activation='softmax'))

    #Adam Optimizer, and Sparse Crosscategorical Entropy as loss function for the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #Train the model
    history = model.fit(X_train,
                        y_train,
                        epochs=500,
                        batch_size=32)

    print("Evaluating model with validation data:")
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print('test_acc: ', test_acc)

    #Predict the validation set
    predictions = model.predict(X_val)
    predicted_array = []

    #If you want to visualize record by record, uncomment the print statement
    for i in range(0,len(predictions)):
        predicted = np.argmax(predictions[i])
        predicted_array.append(predicted)
        expected = np.array(y_val)
        expected = expected[i]
        correct = predicted == expected
        #print("Predicted value: " + str(predicted) + ", Expected value: " + str(expected) + ", Correct: " + str(correct))

    #Generate confusion matrix and save it as a csv file
    conf_mat = tf.math.confusion_matrix(
        labels=np.array(y_val), predictions=predicted_array, num_classes=6
    ).numpy()
    np.savetxt("confusion.csv", conf_mat, delimiter=",")

    #Open the test data file
    testdata = pd.read_csv('testdata.csv')

    # Drop unnecessary columns
    testdata_raw = testdata.drop(['filename'], axis=1)

    # Normalize test data
    scaler = StandardScaler()
    X_test = scaler.fit_transform(np.array(testdata_raw.iloc[:, :-1], dtype=float))

    # Predict the test data
    test_predictions = model.predict(X_test)

    # Generate predictions.csv file to submit to Kaggle
    test_result = []

    for test_prediction in test_predictions:
        test_result.append(np.argmax(test_prediction))

    prediction_df = pd.DataFrame(zip(testdata.filename,test_result))

    prediction_df.rename(columns={0: 'id', 1: 'genre'},inplace=True)

    prediction_df.to_csv('predictions.csv', index=False)


def EnsembleNNModel(n):

    # This stores both the total results, and partial results
    total_result = []

    results_list = []

    i = 0

    # N corresponds to the number of neural networks in the Ensemble
    while i < n:

        # Open csv file
        data = pd.read_csv('data.csv')

        # Bagging on instances
        data = data.sample(frac=0.8, replace=False)

        print("Creating Neural Network #" + str(i+1) + " for Ensemble")

        # Drop unnecessary columns
        columns = ['filename','label']
        genre_list = data.iloc[:, -1]
        y = genre_list

        # Bagging on attributes
        data = data.drop(columns, axis=1)
        data = data.sample(frac=0.8, axis='columns')

        # Normalizing data
        scaler = StandardScaler()
        X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

        # Splitting data into training and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

        # Randomize parameters that determine the architecture
        model = models.Sequential()

        # 5 to 8 layers
        n_layers = r.randint(5,8)

        # 256 to 512 neurons on first layer
        neurons = r.randint(2,4)*(128)

        # Keep dropout layer constant
        dropout_rate = 0.2

        # 3 possible activation functions
        act_i = r.randint(0,2)
        act_arr = ['relu','LeakyReLU','sigmoid']

        # Create input layer
        model.add(layers.Dense(neurons, activation=act_arr[act_i], input_shape=(X_train.shape[1],)))
        model.add(layers.Dropout(dropout_rate))

        # Generate hidden layers randomly

        for k in range(0, n_layers):
            if neurons > 12:
                neurons = neurons // 2
            act_i = r.randint(0, 1)
            act_arr = ['relu', 'LeakyReLU']
            model.add(layers.Dense(neurons, activation=act_arr[act_i],
                                   kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                   bias_regularizer=regularizers.L2(1e-4),
                                   activity_regularizer=regularizers.L2(1e-5)
                                   ))
            model.add(layers.Dropout(dropout_rate))

        # Output layer

        model.add(layers.Dense(6, activation='softmax'))

        # Compile model with adam optimizer

        model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

        history = model.fit(X_train,
                        y_train,
                        epochs=100,
                        batch_size=64)

        # Validate the model

        print("Evaluating model with test data:")
        test_loss, test_acc = model.evaluate(X_val, y_val)
        print('test_acc: ', test_acc)

        predictions = model.predict(X_val)

        # If you want to visualize how records are classified one by one, uncomment this section.

        for j in range(0,len(predictions)):
            predicted = np.argmax(predictions[i])
            expected = np.array(y_val)
            expected = expected[i]
            correct = predicted == expected
            #print("Predicted value: " + str(predicted) + ", Expected value: " + str(expected) + ", Correct: " + str(correct))

        testdata = pd.read_csv('testdata.csv')

        # Drop unnecessary columns
        testdata_raw = testdata.drop(['filename'], axis=1)
        testdata_raw = testdata_raw.filter(data.columns)

        # Normalize test data
        scaler = StandardScaler()
        X_test = scaler.fit_transform(np.array(testdata_raw.iloc[:, :-1], dtype=float))

        # Predict the test data
        test_predictions = model.predict(X_test)

        test_result = []

        for test_prediction in test_predictions:
            test_result.append(np.argmax(test_prediction))

        # We only add NN with accuracy of more than 0.5 to the ensemble
        # It prevents problems like Gradient Vanishing in case of sigmoid
        if test_acc > 0.5:
            if i == 0:
                results_list = pd.DataFrame({'NN' + str(i): test_result})
                print(results_list)
            else:
                new_result = pd.DataFrame({'NN' + str(i): test_result})
                results_list = results_list.join(new_result)
                print(results_list)
            if i == n-1:
                model_result = results_list.mode(axis='columns')
                total_result = pd.Series(model_result[0])
            i = i + 1

    testdata = pd.read_csv('testdata.csv')

    prediction_df = pd.DataFrame(zip(testdata.filename,total_result.astype(int)))

    prediction_df.rename(columns={0: 'id', 1: 'genre'},inplace=True)

    # Final Ensemble Prediction
    print(prediction_df)

    prediction_df.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    NNModel()
    #EnsembleNNModel(10)