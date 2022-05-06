import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

from utils import get_classes
from sklearn.preprocessing import StandardScaler


def get_single_mfcc_features(file_ids, directory):
  """return a matrix consisting of a single mfcc for each example in the classification array"""
  sc = StandardScaler()

  features = np.zeros((len(file_ids), 1515))
  for i in range(len(file_ids)):
    filepath = os.path.join(directory, file_ids[i])
    pcm_data, sr = librosa.load(filepath, offset=15.0, duration=60.0)
    n_fft = int(sr * 0.02)
    hop_length = n_fft // 2

    mfccs = librosa.feature.mfcc(y=pcm_data, sr=sr, n_mfcc=1, n_fft=n_fft,
                                 hop_length=hop_length)

    mfccs = mfccs.transpose()

    features[i, 0:len(mfccs)] = sc.fit_transform(mfccs.reshape(-1, 1)).ravel()
  return features


def vizualize_individual_mfccs(file_list, directory):
  genres = {
    '0': "Rock",
    '1': "Pop",
    '2': "Folk",
    '3': "Instrumental",
    '4': "Electronic",
    '5': "Hip-Hop"
  }

  columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

  for i in range(len(file_list)):
    g = file_list[i, 1]
    title = "{0:s} MFCCs".format(genres[g])

    filepath = os.path.join(directory, file_list[i, 0])
    pcm_data, sr = librosa.load(filepath, offset=15.0, duration=60.0)
    n_fft = int(sr * 0.02)
    hop_length = n_fft // 2

    # print(np.linalg.norm(test))

    mfccs = librosa.feature.mfcc(y=pcm_data, sr=sr, n_mfcc=13, n_fft=n_fft,
                                 hop_length=hop_length)
    mfcc_data = np.swapaxes(mfccs, 0, 1)

    df = pd.DataFrame(mfcc_data, columns=columns)

    plt.plot(df['0'], label='MFCC 0', color='crimson')
    plt.plot(df['1'], label='MFCC 1', color='c')
    plt.plot(df['2'], label='MFCC 2', color='skyblue')
    plt.plot(df['3'], label='MFCC 3', color='dodgerblue')
    plt.plot(df['4'], label='MFCC 4', color='slategrey')
    plt.plot(df['5'], label='MFCC 5', color='darkblue')
    plt.plot(df['6'], label='MFCC 6', color='slateblue')
    plt.plot(df['7'], label='MFCC 7', color='rebeccapurple')
    plt.plot(df['8'], label='MFCC 8', color='darkviolet')
    plt.plot(df['9'], label='MFCC 9', color='violet')
    plt.plot(df['10'], label='MFCC 10', color='fuchsia')
    plt.plot(df['11'], label='MFCC 11', color='deeppink')
    plt.plot(df['12'], label='MFCC 12', color='cadetblue')

    plt.title(title)
    plt.show()

def scatter_by_amp(file_list, directory):
  df = pd.DataFrame(file_list, columns=['id', 'genre'])
  amplitude = []
  means = []

  # Load the audio
  for i in range(len(file_list)):

    filepath = os.path.join(directory, file_list[i][0])
    pcm_data, sr = librosa.load(filepath, offset=15.0, duration=60.0)
    n_fft = int(sr * 0.02)
    hop_length = n_fft // 2
    mfccs = librosa.feature.mfcc(
        y=pcm_data, sr=sr, n_mfcc=1, n_fft=n_fft, hop_length=hop_length
    )
    amplitude.append(np.linalg.norm(mfccs))
    freq = librosa.mel_frequencies(n_mels=40)
    means.append(abs(np.mean(mfccs)))

  df['amplitude'] = amplitude
  df['means'] = means

  grouped = df.groupby("genre")

  fig, ax = plt.subplots()

  colors = {
    '0': "deeppink",
    '1': "greenyellow",
    '2': "orange",
    '3': "blueviolet",
    '4': "thistle",
    '5': "deepskyblue",
  }

  for key, group in grouped:
    group.plot(
        ax=ax, kind="scatter", x='means', y='amplitude', label=key, color=colors[key]
    )
  plt.show()



def accumulate_single_mfcc(directory, classifications):
  """ creates a scatter plot of mfcc's using only n_mfcc=1
    plot should be either maxes, mins, or ranges
    classifications should be 2d array with id and genre
  """ ""

  maxes = []
  mins = []
  means = []
  ranges = []
  clses = []

  # Load the audio
  for i in range(len(classifications)):
    genre = classifications[i][1]
    clses.append(int(genre))

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
  maxes, mins, means, ranges, clses = accumulate_single_mfcc(directory,
                                                             classifications)

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


def get_processed_mfccs(directory, file_ids):
  """generates mfccs and the 'smooths' the data returning a n*72 matrix of attributes"""

  mfcc_size = 13
  sample_rate = 44100

  data = np.zeros((1, mfcc_size * 3))

  # Load the audio
  for file in file_ids:

    filepath = os.path.join(directory, file)
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
  file_list = get_classes(r"D:\proj3_data\project3\train.csv")
  # classifications = get_classes(r"D:\repos\CS529_Project3\train1.csv")
  # file_list = get_classes(r"D:\repos\CS529_Project3\examples.csv")
  file_list = np.array(file_list)
  print(file_list.shape)
  train_dir = r"D:\proj3_data\project3\trainwav"
  # vizualize_individual_mfccs(file_list, train_dir)
  scatter_by_amp(file_list, train_dir)
  # plot_single_mfcc("ranges", train_dir, classifications)
  # x = get_processed_mfccs(train_dir, classifications)
  # z = get_single_mfcc_features(classifications, train_dir)
