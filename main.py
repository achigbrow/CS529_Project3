import tf_test as tf

import librosa as lib

import librosa.display

import IPython.display as ipd

import matplotlib.pyplot as plt


def audio_features():
    audio_path = './data/train/00907299.mp3'
    x, sr = lib.load(audio_path)
    ipd.Audio(audio_path)
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(x, sr=sr)


if __name__ == "__main__":
    #tf.tf_test()
    audio_features()
