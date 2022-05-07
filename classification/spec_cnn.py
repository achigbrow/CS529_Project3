import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import tensorflow as tf
import time

from visualization.spectrogram import ms_features
from utils import get_classes


def get_training(filepath):
    train_df = pd.read_csv(filepath)
    train_df.columns = ["id", "genre"]
    train_df.head()
    ids = train_df["id"]
    padded = []
    wavs = []

    for id in ids:
        pad = str(id).zfill(8)
        padded.append(pad)
        wavs.append("{0:s}.wav".format(pad))

    train_df["padded"] = padded
    train_df["wav"] = wavs

    train_df.head()

    return train_df


def get_test(filepath):
    test_df = pd.read_csv(filepath)
    test_df.columns = ["id"]
    ids = test_df["id"]
    wavs = []
    padded = []

    for id in ids:
        pad = str(id).zfill(8)
        padded.append(pad)
        wavs.append("{0:s}.wav".format(pad))

    test_df["padded"] = padded
    test_df["wav"] = wavs

    test_df.head()
    return test_df


def run_model(X_train, y_train, X_val, y_val, X_test, y_test, X_submit):
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    CNNmodel = models.Sequential()
    CNNmodel.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape))
    CNNmodel.add(layers.MaxPooling2D((2, 4)))
    CNNmodel.add(layers.Conv2D(64, (3, 5), activation="relu"))
    CNNmodel.add(layers.MaxPooling2D((2, 4)))
    CNNmodel.add(layers.Dropout(0.2))
    CNNmodel.add(layers.Flatten())
    CNNmodel.add(layers.Dense(32, activation="relu"))
    CNNmodel.add(layers.Dense(6, activation="softmax"))
    CNNmodel.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    history = CNNmodel.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
    predicted = CNNmodel.predict(X_test)

    predicted = np.argmax(predicted, axis=1)

    conf_mat = tf.math.confusion_matrix(
        labels=y_test, predictions=predicted, num_classes=6
    ).numpy()
    np.savetxt("confusion.csv", conf_mat, delimiter=",")

    # check accuracy
    history_dict = history.history
    loss_values = history_dict["loss"]
    acc_values = history_dict["accuracy"]
    val_loss_values = history_dict["val_loss"]
    val_acc_values = history_dict["val_accuracy"]
    epochs = range(1, 21)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(epochs, loss_values, "bo", label="Training Loss")
    ax1.plot(epochs, val_loss_values, "orange", label="Validation Loss")
    ax1.set_title("Training and validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(epochs, acc_values, "bo", label="Training accuracy")
    ax2.plot(epochs, val_acc_values, "orange", label="Validation accuracy")
    ax2.set_title("Training and validation accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    plt.show()

    submission = CNNmodel.predict(X_submit)
    return np.argmax(submission, axis=1)


def driver():
    train_fp = r"D:\proj3_data\project3\train.csv"
    # train_fp = r"D:\repos\CS529_Project3\train1.csv"
    test_fp = r"D:\proj3_data\project3\new_test_idx.csv"
    # test_fp = r"D:\repos\CS529_Project3\test1.csv"

    # train_list = np.array(get_classes(train_fp))
    # test_list = np.array(get_classes(test_fp))

    example_list = np.array(get_classes(r"D:\repos\CS529_Project3\examples.csv"))

    train_df = get_training(train_fp)
    test_df = get_test(test_fp)

    train_dir = r"D:\proj3_data\project3\trainwav"
    test_dir = r"D:\proj3_data\project3\testwav"

    X = train_df.drop("genre", axis=1)
    y = train_df.genre


    # Split once to get the test and training set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )
    print(X_train.shape, X_test.shape)

    # Split twice to get the validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=123
    )
    print(
        X_train.shape, X_test.shape, X_val.shape, len(y_train), len(y_test), len(y_val)
    )

    print("getting test features")
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    test_features, test_labels = ms_features(
        pd.concat([X_test, y_test], axis=1), train_dir
    )

    print("getting validation features")
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    val_features, val_labels = ms_features(pd.concat([X_val, y_val], axis=1), train_dir)

    print("getting training features")
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    train_features, train_labels = ms_features(
        pd.concat([X_train, y_train], axis=1), train_dir
    )

    print("getting submission features")
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    submit_features = ms_features(test_df, test_dir, False)

    print("log")
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)

    train_features = np.log2(train_features + 1)
    train_features = train_features.reshape(train_features.shape[0], train_features.shape[1], train_features.shape[2], 1)
    print(train_features.shape)
    np.save("train_features", train_features)

    test_features = np.log2(test_features + 1)
    test_features = test_features.reshape(test_features.shape[0], test_features.shape[1], test_features.shape[2], 1)
    np.save("test_features", test_features)

    val_features = np.log2(val_features + 1)
    val_features = val_features.reshape(val_features.shape[0], val_features.shape[1], val_features.shape[2], 1)
    np.save("val_features", val_features)

    submit_features = np.log2(submit_features + 1)
    submit_features = submit_features.reshape(submit_features.shape[0], submit_features.shape[1], submit_features.shape[2], 1)
    np.save("submit_features", test_features)

    train_labels = np.array(train_labels)
    np.save("train_labels", train_labels)

    test_labels = np.array(test_labels)
    np.save("test_labels", test_labels)

    val_labels = np.array(val_labels)
    np.save("val_labels", val_labels)

    print("getting predictions")
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    print(test_features.shape, val_features.shape, train_features.shape, submit_features.shape)
    print(test_labels.shape, val_labels.shape, train_labels.shape)
    predictions = run_model(
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        submit_features,
    )
    print("done")
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)

    submit_df = pd.DataFrame({"id": test_df["id"], "genre": predictions})
    submit_df.to_csv("submitCNN.csv", index=False)


if __name__ == "__main__":
    driver()
