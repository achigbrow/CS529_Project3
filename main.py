import pandas as pd
import tf_test as tf

import librosa as lib

import librosa.display

import IPython.display as ipd

import matplotlib.pyplot as plt
import numpy as np

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
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train,
                        y_train,
                        epochs=100,
                        batch_size=128)

    print("Evaluating model with test data:")
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print('test_acc: ', test_acc)

    predictions = model.predict(X_val)

    for i in range(0,len(predictions)):
        predicted = np.argmax(predictions[i])
        expected = np.array(y_val)
        expected = expected[i]
        correct = predicted == expected
        print("Predicted value: " + str(predicted) + ", Expected value: " + str(expected) + ", Correct: " + str(correct))

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

    print(prediction_df)

    prediction_df.to_csv('predictions.csv', index=False)



if __name__ == "__main__":
    #tf.tf_test()
    NNModel()
