import pandas as pd
import tf_test as tft
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


def NNModel():
    data = pd.read_csv('data.csv')

    #Drop unnecessary column
    data = data.drop(['filename'], axis=1)
    #print(data.head())

    genre_list = data.iloc[:, -1]
    y = genre_list
    #print(y)

    #normalizing
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
    #print(X)


    #Splitting data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

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

    sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)

    model.compile(optimizer=sgd,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train,
                        y_train,
                        epochs=500,
                        batch_size=32)

    print("Evaluating model with test data:")
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print('test_acc: ', test_acc)

    predictions = model.predict(X_val)

    for i in range(0,len(predictions)):
        predicted = np.argmax(predictions[i])
        expected = np.array(y_val)
        expected = expected[i]
        correct = predicted == expected
        #print("Predicted value: " + str(predicted) + ", Expected value: " + str(expected) + ", Correct: " + str(correct))

    #Predict the test data
    testdata = pd.read_csv('testdata.csv')

    # Drop unnecessary column
    testdata_raw = testdata.drop(['filename'], axis=1)

    # normalizing
    scaler = StandardScaler()
    X_test = scaler.fit_transform(np.array(testdata_raw.iloc[:, :-1], dtype=float))
    #print(X_test)

    test_predictions = model.predict(X_test)

    test_result = []

    for test_prediction in test_predictions:
        test_result.append(np.argmax(test_prediction))

    prediction_df = pd.DataFrame(zip(testdata.filename,test_result))

    prediction_df.rename(columns={0: 'id', 1: 'genre'},inplace=True)

    #print(prediction_df)

    prediction_df.to_csv('predictions.csv', index=False)


def EnsembleNNModel(n):
    total_result = []

    results_list = []

    i = 0

    while i < n:
        data = pd.read_csv('data.csv')
        data = data.sample(frac=1.0, replace=True)

        print("Creating Neural Network #" + str(i+1) + " for Ensemble")

        #Drop unnecessary column
        columns = ['filename','label']
        genre_list = data.iloc[:, -1]
        y = genre_list

        data = data.drop(columns, axis=1)
        data = data.sample(frac=0.8, axis='columns')

        #normalizing
        scaler = StandardScaler()
        X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
        #print(X)

        #Splitting data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        model = models.Sequential()

        n_layers = r.randint(5,8)
        neurons = r.randint(1,8)*(128)
        dropout_rate = 0.2
        act_i = r.randint(0,1)
        act_arr = ['relu','LeakyReLU']

        model.add(layers.Dense(neurons, activation=act_arr[act_i], input_shape=(X_train.shape[1],)))
        model.add(layers.Dropout(dropout_rate))

        for k in range(0, n_layers):
            if neurons > 12:
                neurons = neurons // 2
            act_i = r.randint(0, 1)
            act_arr = ['relu', 'LeakyReLU']
            model.add(layers.Dense(neurons, activation=act_arr[act_i]))
            model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(6, activation='softmax'))

        #lr = r.randint(1,5)*0.01
        #d = r.randint(1,5)*1e-6
        #m = r.randint(1,5)*0.1
        #sgd = tf.keras.optimizers.SGD(learning_rate=lr, decay=d, momentum=m, nesterov=True)

        #optimizer = ['adam',sgd]
        #o_index = r.randint(0,1)

        model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

        history = model.fit(X_train,
                        y_train,
                        epochs=100,
                        batch_size=64)

        print("Evaluating model with test data:")
        test_loss, test_acc = model.evaluate(X_val, y_val)
        print('test_acc: ', test_acc)

        predictions = model.predict(X_val)

        for j in range(0,len(predictions)):
            predicted = np.argmax(predictions[i])
            expected = np.array(y_val)
            expected = expected[i]
            correct = predicted == expected
            #print("Predicted value: " + str(predicted) + ", Expected value: " + str(expected) + ", Correct: " + str(correct))

        testdata = pd.read_csv('testdata.csv')

        # Drop unnecessary column
        testdata_raw = testdata.drop(['filename'], axis=1)
        testdata_raw = testdata_raw.filter(data.columns)

        # normalizing
        scaler = StandardScaler()
        X_test = scaler.fit_transform(np.array(testdata_raw.iloc[:, :-1], dtype=float))
        #print(X_test)

        test_predictions = model.predict(X_test)

        test_result = []

        for test_prediction in test_predictions:
            test_result.append(np.argmax(test_prediction))

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

    print(prediction_df)

    prediction_df.to_csv('predictions.csv', index=False)



if __name__ == "__main__":
    #tft.tf_test()
    #NNModel()
    EnsembleNNModel(10)