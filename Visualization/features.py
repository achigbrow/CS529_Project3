import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.preprocessing import normalize


def get_mfccs(directory, classifications):
    sample_rate = 44100
    mfcc_size = 13

    sample_count = len(classifications)
    print(sample_count)

    data = np.zeros([1, (mfcc_size*3) + 1])

    # Load the audio
    for i in range(len(classifications)):
        cls = np.array(classifications[i][1])

        filepath = os.path.join(directory, classifications[i][0])
        pcm_data, _ = librosa.load(filepath)


        # Compute a vector of n * 13 mfccs
        mfccs = librosa.feature.mfcc(y=pcm_data, sr=sample_rate, n_mfcc=mfcc_size)

        nrows, ncols = mfccs.shape

        mfccs = np.transpose(mfccs)
        # Compress data by analyzing numbers

        # Get the standard deviation
        stddev_features = np.std(mfccs, axis=0)


        # Get the mean
        mean_features = np.mean(mfccs, axis=0)

        # Get the average difference of the features
        average_difference_features = np.zeros((nrows,))
        for i in range(0, len(mfccs) - 2, 2):
            test = mfccs[i] - mfccs[i + 1]
            average_difference_features += test

        average_difference_features /= len(mfccs) // 2
        average_difference_features = np.array(average_difference_features)

        # Concatenate the features to a single feature vector
        concat_features_features = np.hstack((stddev_features, mean_features))
        concat_features_features = np.hstack(
            (concat_features_features, average_difference_features)
        )
        norm = np.linalg.norm(concat_features_features)
        concat_features_features = concat_features_features / norm
        concat_features_features = np.hstack((concat_features_features, cls))
        data = np.vstack([data, concat_features_features])

    print(data.shape)
    data = data[1:, :]
    print(data.shape)

    x = np.transpose(data[:, -1])
    print(x.shape)
    features = data[:, :-1]
    features = features.astype(float)
    print(features.dtype)
    print(features.shape)
    y = np.median(features, axis=1)
    print(y.shape)

    plt.scatter(x, y, c="blue")
    plt.show()

    return data


if __name__ == "__main__":
    with open(r"D:\proj3_data\project3\train.csv") as f:
        reader = csv.reader(f)
        data = list(reader)

    for i in range(len(data)):
        temp = f"{int(data[i][0]):08d}"
        data[i][0] = "{0:s}.wav".format(temp)

    print(data[0])
    get_mfccs(r"D:\proj3_data\project3\trainwav", data)
