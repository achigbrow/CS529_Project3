import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_classes
from sklearn.preprocessing import StandardScaler

def get_single_mfcc_features(classification, directory):
    """return a matrix consisting of a single mfcc for each example in the classification array"""
    sc = StandardScaler()

    features = np.zeros((len(classification), 1515))
    for i in range(len(classification)):

        filepath = os.path.join(directory, classification[i][0])
        pcm_data, sr = librosa.load(filepath, offset=15.0, duration=60.0)
        n_fft = int(sr * 0.02)
        hop_length = n_fft // 2

        mfccs = librosa.feature.mfcc(y=pcm_data, sr=sr, n_mfcc=1, n_fft=n_fft, hop_length=hop_length)

        mfccs = mfccs.transpose()

        features[i, 0:len(mfccs)] = sc.fit_transform(mfccs.reshape(-1, 1)).ravel()
    return features

def accumulate_single_mfcc(directory, classifications):
    """ creates a scatter plot of mfcc's using only n_mfcc=1
    plot should be either maxes, mins, or ranges
  """ ""

    maxes = []
    mins = []
    means = []
    ranges = []
    clses = []

    # Load the audio
    for i in range(len(classifications)):
        cls = classifications[i][1]
        clses.append(int(cls))

        filepath = os.path.join(directory, classifications[i][0])
        pcm_data, sr = librosa.load(filepath)
        n_fft = int(sr * 0.02)
        hop_length = n_fft // 2
        mfccs = librosa.feature.mfcc(
            y=pcm_data, sr=sr, n_mfcc=1, n_fft=n_fft, hop_length=hop_length
        )

        mfccs = np.transpose(mfccs)

        maxes.append(np.max(mfccs))
        mins.append(np.min(mfccs))
        means.append(np.abs((np.mean(mfccs))))
        ranges.append(np.abs(np.max(mfccs)) + np.abs(np.min(mfccs)))

    return maxes, mins, means, ranges, clses


def plot_single_mfcc(plot, directory, classifications):
    maxes, mins, means, ranges, clses = accumulate_single_mfcc(directory, classifications)

    df = pd.DataFrame(
        {
            "maxes": maxes,
            "mins": mins,
            "means": means,
            "ranges": ranges,
            "classes": clses,
        }
    )

    grouped = df.groupby("classes")

    fig, ax = plt.subplots()

    colors = {
        0: "deeppink",
        1: "greenyellow",
        2: "orange",
        3: "blueviolet",
        4: "thistle",
        5: "deepskyblue",
    }

    for key, group in grouped:
        group.plot(
            ax=ax, kind="scatter", x="means", y=plot, label=key, color=colors[key]
        )
    plt.show()


def get_processed_mfccs(directory, classifications):
    """generates mfccs and the 'smooths' the data returning a n*72 matrix of attributes"""

    mfcc_size = 24
    sample_rate = 44100

    data = np.zeros((1, mfcc_size * 3))

    # Load the audio
    for i in range(len(classifications)):

        filepath = os.path.join(directory, classifications[i])
        pcm_data, _ = librosa.load(filepath, sr=sample_rate)
        n_fft = int(sample_rate * 0.02)
        hop_length = n_fft // 2
        mfccs = librosa.feature.mfcc(
            y=pcm_data,
            sr=sample_rate,
            n_mfcc=mfcc_size,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        nrows, ncols = mfccs.shape
        mfccs = mfccs.transpose()

        # Compress data by analyzing numbers

        # Get the standard deviation
        stddev_features = np.std(mfccs, axis=0)
        # Get the mean
        mean_features = np.mean(mfccs, axis=0)

        # Get the average difference of the features
        average_difference_features = np.zeros((nrows,))
        for i in range(0, ncols - 2, 2):
            test = mfccs[i, :] - mfccs[i + 1, :]

            average_difference_features += test

            average_difference_features /= len(mfccs) // 2
            average_difference_features = np.array(average_difference_features)

        # Concatenate the features to a single feature vector
        concat_features_features = np.hstack((stddev_features, mean_features))
        concat_features_features = np.hstack(
            (concat_features_features, average_difference_features)
        )
        data = np.vstack([data, concat_features_features])

    data = data[1:, :]
    return data


if __name__ == "__main__":
    classifications = get_classes(r"D:\proj3_data\project3\train.csv")
    # classifications = get_classes(r"D:\repos\CS529_Project3\train1.csv")
    train_dir = r"D:\proj3_data\project3\trainwav"
    plot_single_mfcc("ranges", train_dir, classifications)
    # x = get_processed_mfccs(train_dir, classifications)
    # z = get_single_mfcc_features(classifications, train_dir)

