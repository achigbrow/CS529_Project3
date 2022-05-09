# see the readme for a list of references used to build this class
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import os

from random import randint


def display_spectrogram(file_list, directory):
    """displays a plot of the mel-spectrogram of each file in the list

    :param file_list: list of .wav file names and their genre
    :param directory: where .wav files are located
    :return: none
    """
    genres = {
        "0": "Rock",
        "1": "Pop",
        "2": "Folk",
        "3": "Instrumental",
        "4": "Electronic",
        "5": "Hip-Hop",
    }

    for i in range(len(file_list)):
        g = file_list[i, 1]
        title = "{0:s} Melspectrogram".format(genres[g])
        n_fft = 500
        hop_length = round(n_fft / 2)
        filepath = os.path.join(directory, file_list[i, 0])
        # load 3 second time series
        y, sr = librosa.load(filepath, offset=15.0, duration=3)
        # obtain the mel-spectrogram array
        spectros = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=64,
            fmax=8000,
        )

        fig, ax = plt.subplots()
        s_db = librosa.power_to_db(spectros, ref=np.max)
        img = librosa.display.specshow(
            s_db, x_axis="time", y_axis="mel", sr=sr, fmax=8000, ax=ax
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set(title=title)
        plt.show()


def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h = array.shape[0]
    w = array.shape[1]
    a = max((xx - h) // 2, 0)
    aa = max(0, xx - a - h)
    b = max(0, (yy - w) // 2)
    bb = max(yy - b - w, 0)
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode="constant")


def ms_features(file_df, directory, genre=True):
    """Builds a padded array of melspectrograms

    :param file_df: dataframe containing the file data
    :param directory: where the .wav files live
    :param genre: whether or not to return a list of genres
    :return: either a feature array and a list of genres or just a feature array
    """
    genres = []
    features = []
    n_fft = 264
    hop_length = round(n_fft / 2)
    for index, row in file_df.iterrows():
        filename = row.wav
        filepath = "{0:s}/{1:s}".format(directory, filename)
        offset = randint(1, 20)
        if genre:
            genres.append(file_df["genre"][index])
        # load 3 second time series
        y, sr = librosa.load(filepath, offset=offset, duration=3.0)
        # obtain mel-spectrograms
        spectros = padding(librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, fmax=8000, n_fft=n_fft, hop_length=hop_length
        ), 66, 550)

        features.append(spectros[np.newaxis, ...])
    output = np.concatenate(features, axis=0)

    if genre:
        return (np.array(output), genres)
    else:
        return np.array(output)


if __name__ == "__main__":
    print("try calling this with the correct file")
