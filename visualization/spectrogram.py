from utils import get_classes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import librosa.display
import os

from random import randint

def display_spectrogram(file_list, directory):
  genres = {
    '0': "Rock",
    '1': "Pop",
    '2': "Folk",
    '3': "Instrumental",
    '4': "Electronic",
    '5': "Hip-Hop"
  }

  for i in range(len(file_list)):
    g = file_list[i, 1]
    title = "{0:s} Melspectrogram".format(genres[g])
    n_fft = 500
    hop_length = round(n_fft/2)
    filepath = os.path.join(directory, file_list[i, 0])
    y, sr = librosa.load(filepath, offset=15.0, duration=3)
    spectros = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64,
                                              fmax=8000,)
    print(spectros.shape)

    fig, ax = plt.subplots()
    s_db = librosa.power_to_db(spectros, ref=np.max)
    img = librosa.display.specshow(s_db, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
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
  genres = []
  features = []
  for index, row in file_df.iterrows():
    filename = row.wav
    filepath = "{0:s}/{1:s}".format(directory, filename)
    offset = randint(1, 20)
    if genre:
      genres.append(file_df['genre'][index])

    y, sr = librosa.load(filepath, offset=offset, duration=3.0)
    spectros = padding(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64,
                                              fmax=8000), 64, 135)
    # print(spectros.shape)
    features.append(spectros[np.newaxis, ...])
  output = np.concatenate(features, axis=0)

  if genre:
    return (np.array(output), genres)
  else:
    return np.array(output)


if __name__ == "__main__":
  print("whoops")