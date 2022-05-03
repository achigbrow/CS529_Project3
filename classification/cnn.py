import numpy
import pandas
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import tensorflow as tf
import time


def get_training(filepath):
  train_df = pd.read_csv(filepath)
  train_df.columns = ['id', 'genre']
  train_df.head()
  ids = train_df['id']
  wavs = []
  padded = []

  for id in ids:
    pad = str(id).zfill(8)
    padded.append(pad)
    wavs.append("{0:s}.wav".format(pad))

  train_df['padded'] = padded
  train_df['wav'] = wavs

  train_df.head()

  return train_df


def get_test(filepath):
  test_df = pd.read_csv(filepath)
  test_df.columns = ['id']
  ids = test_df['id']
  wavs = []
  padded = []

  for id in ids:
    pad = str(id).zfill(8)
    padded.append(pad)
    wavs.append("{0:s}.wav".format(pad))

  test_df['padded'] = padded
  test_df['wav'] = wavs

  test_df.head()
  return test_df


def display_wavplot(directory, df, sample_n):
  filename = df.wav[sample_n]
  filepath = '{0:s}/{1:s}'.format(directory, filename)
  fig, ax = plt.subplots(nrows=3, sharex=True)
  y, sr = librosa.load(filepath)  # load the file

  librosa.display.waveshow(y, sr=sr, x_axis='time', color='cyan', ax=ax[1])
  ax[1].set(title='Envelope view, stereo')
  ax[1].label_outer()


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
  return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def generate_features(y, sr):
  max_size = 2000  # my max audio file feature width
  hop_length = 512
  n_fft = 255
  stft = padding(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)),
                 128, max_size)
  MFCCs = padding(
    librosa.feature.mfcc(y, n_fft=n_fft, hop_length=hop_length, n_mfcc=128),
    128, max_size)
  spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
  chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
  spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
  # Now the padding part
  image = np.array([padding(normalize(spec_bw), 1, max_size)]).reshape(1,
                                                                       max_size)
  image = np.append(image, padding(normalize(spec_centroid), 1, max_size),
                    axis=0)
  # repeat the padded spec_bw,spec_centroid and chroma stft until they are stft and MFCC-sized
  for i in range(0, 9):
    image = np.append(image, padding(normalize(spec_bw), 1, max_size), axis=0)
    image = np.append(image, padding(normalize(spec_centroid), 1, max_size),
                      axis=0)
    image = np.append(image, padding(normalize(chroma_stft), 12, max_size),
                      axis=0)
  image = np.dstack((image, np.abs(stft)))
  image = np.dstack((image, MFCCs))
  return image


def get_features(df_in, directory):
  features = []
  labels = []  # empty array to store labels
  # For each species, determine how many augmentations are needed
  df_in = df_in.reset_index()
  for i in df_in.genre.unique():
    # all the file indices with the same genre
    filelist = df_in.loc[df_in.genre == i].index
    for j in range(0, len(filelist)):  # len(filelist)
      filename = df_in.iloc[filelist[j]].wav
      file_path = "{0:s}/{1:s}".format(directory, filename)
      genre = i
      # Load the file
      y, sr = librosa.load(file_path, sr=28000)
      # cut the file to signal start and end
      # generate features & output numpy array
      data = generate_features(y, sr)
      features.append(data[np.newaxis, ...])
      labels.append(genre)
  output = np.concatenate(features, axis=0)
  return (np.array(output), labels)


def get_submit_features(df_in, directory):
  features = []
  for index, row in df_in.iterrows():
    filename = row.wav
    file_path = "{0:s}/{1:s}".format(directory, filename)
    y, sr = librosa.load(file_path, sr=28000)
    # generate features & output numpy array
    data = generate_features(y, sr)
    features.append(data[np.newaxis, ...])
  output = np.concatenate(features, axis=0)
  return np.array(output)


def run_model(X_train, y_train, X_val, y_val, X_test, y_test, X_submit):
  input_shape = (128, 2000, 3)
  CNNmodel = models.Sequential()
  CNNmodel.add(
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  CNNmodel.add(layers.MaxPooling2D((2, 2)))
  CNNmodel.add(layers.Dropout(0.2))
  CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
  CNNmodel.add(layers.MaxPooling2D((2, 2)))
  CNNmodel.add(layers.Dropout(0.2))
  CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
  CNNmodel.add(layers.Flatten())
  CNNmodel.add(layers.Dense(64, activation='relu'))
  CNNmodel.add(layers.Dropout(0.2))
  CNNmodel.add(layers.Dense(32, activation='relu'))
  CNNmodel.add(layers.Dense(6, activation='softmax'))
  CNNmodel.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(
                     from_logits=False), metrics=['accuracy'])
  history = CNNmodel.fit(X_train, y_train, epochs=20,
                         validation_data=(X_val, y_val))
  predicted = CNNmodel.predict(X_test)

  predicted = numpy.argmax(predicted, axis=1)


  conf_mat = tf.math.confusion_matrix(labels=y_test, predictions=predicted,
                                      num_classes=6).numpy()
  numpy.savetxt('confusion.csv', conf_mat, delimiter=",")

  # check accuracy
  history_dict = history.history
  loss_values = history_dict['loss']
  acc_values = history_dict['accuracy']
  val_loss_values = history_dict['val_loss']
  val_acc_values = history_dict['val_accuracy']
  epochs = range(1, 21)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
  ax1.plot(epochs, loss_values, 'bo', label='Training Loss')
  ax1.plot(epochs, val_loss_values, 'orange', label='Validation Loss')
  ax1.set_title('Training and validation loss')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Loss')
  ax1.legend()
  ax2.plot(epochs, acc_values, 'bo', label='Training accuracy')
  ax2.plot(epochs, val_acc_values, 'orange', label='Validation accuracy')
  ax2.set_title('Training and validation accuracy')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Accuracy')
  ax2.legend()
  plt.show()

  submission = CNNmodel.predict(X_submit)

  return numpy.argmax(submission, axis=1)


def driver():
  train_fp = r'D:\proj3_data\project3\train.csv'
  # train_fp = r'D:\repos\CS529_Project3\train1.csv'
  test_fp = r'D:\proj3_data\project3\new_test_idx.csv'
  # test_fp = r'D:\repos\CS529_Project3\test1.csv'
  train_df = get_training(train_fp)
  test_df = get_test(test_fp)

  train_dir = r'D:\proj3_data\project3\trainwav'
  test_dir = r'D:\proj3_data\project3\testwav'

  display_wavplot(train_dir, train_df, 2)

  X = train_df.drop('genre', axis=1)
  y = train_df.genre

  # Split once to get the test and training set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                      random_state=123,
                                                      stratify=y)
  # print(X_train.shape, X_test.shape)

  # Split twice to get the validation set
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                    test_size=0.25,
                                                    random_state=123)
  # print(X_train.shape, X_test.shape, X_val.shape, len(y_train), len(y_test),
  #       len(y_val))

  print('getting test features')
  t = time.localtime()
  current_time = time.strftime("%H:%M:%S", t)
  print(current_time)
  test_features, test_labels = get_features(pd.concat([X_test, y_test], axis=1),
                                            train_dir)

  print('getting validation features')
  t = time.localtime()
  current_time = time.strftime("%H:%M:%S", t)
  print(current_time)
  val_features, val_labels = get_features(pd.concat([X_val, y_val], axis=1),
                                          train_dir)

  print('getting training features')
  t = time.localtime()
  current_time = time.strftime("%H:%M:%S", t)
  print(current_time)
  train_features, train_labels = get_features(
    pd.concat([X_train, y_train], axis=1), train_dir)

  print('getting submission features')
  t = time.localtime()
  current_time = time.strftime("%H:%M:%S", t)
  print(current_time)
  submit_features = get_submit_features(test_df, test_dir)

  print('normalizing')
  t = time.localtime()
  current_time = time.strftime("%H:%M:%S", t)
  print(current_time)
  train_features = np.array((train_features - np.min(train_features)) / (
        np.max(train_features) - np.min(train_features)))
  train_features = train_features / np.std(train_features)
  numpy.save("train_features", train_features)

  test_features = np.array((test_features - np.min(test_features)) / (
      np.max(test_features) - np.min(test_features)))
  test_features = test_features / np.std(test_features)
  numpy.save("test_features", test_features)

  val_features = np.array((val_features - np.min(val_features)) / (
        np.max(val_features) - np.min(val_features)))
  val_features = val_features / np.std(val_features)
  numpy.save("val_features", val_features)

  submit_features = np.array((submit_features - np.min(submit_features)) / (
      np.max(submit_features) - np.min(submit_features)))
  submit_features = submit_features / np.std(submit_features)
  numpy.save("submit_features", test_features)

  train_labels = np.array(train_labels)
  numpy.save("train_labels", train_labels)
  test_labels = np.array(test_labels)
  numpy.save("test_labels", test_labels)
  val_labels = np.array(val_labels)
  numpy.save("val_labels", val_labels)


  print('getting predictions')
  t = time.localtime()
  current_time = time.strftime("%H:%M:%S", t)
  print(current_time)
    # print(test_features.shape, val_features.shape, train_features.shape)
  predictions = run_model(train_features, train_labels, val_features,
                          val_labels, test_features, test_labels,
                          submit_features)
  print('done')
  t = time.localtime()
  current_time = time.strftime("%H:%M:%S", t)
  print(current_time)

  submit_df = pandas.DataFrame({'id': test_df['id'], 'genre': predictions})
  submit_df.to_csv("submitCNN.csv", index=False)


if __name__ == "__main__":
  driver()
