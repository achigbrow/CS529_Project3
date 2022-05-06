from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from utils import get_classes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import os


def get_pca(features, n):

    pca = PCA(n_components=n)
    transformed = pca.fit(features).transform(features)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    return scaler.transform(transformed)

def plot_pca(file_list, directory):
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
        title = "{0:s} PCAs".format(genres[g])

        filepath = os.path.join(directory, file_list[i, 0])
        pcm_data, sr = librosa.load(filepath, offset=15.0, duration=60.0)

        pcm_data = pcm_data.reshape(1, -1)
        pca_data = get_pca(pcm_data, 1)

        df = pd.DataFrame(pca_data, columns=columns)

        plt.plot(df['0'], label='PCA 0', color='crimson')
        plt.plot(df['1'], label='PCA 1', color='c')
        plt.plot(df['2'], label='PCA 2', color='skyblue')
        plt.plot(df['3'], label='PCA 3', color='dodgerblue')
        plt.plot(df['4'], label='PCA 4', color='slategrey')
        plt.plot(df['5'], label='PCA 5', color='darkblue')
        plt.plot(df['6'], label='PCA 6', color='slateblue')
        plt.plot(df['7'], label='PCA 7', color='rebeccapurple')
        plt.plot(df['8'], label='PCA 8', color='darkviolet')
        plt.plot(df['9'], label='PCA 9', color='violet')
        plt.plot(df['10'], label='PCA 10', color='fuchsia')
        plt.plot(df['11'], label='PCA 11', color='deeppink')
        plt.plot(df['12'], label='PCA 12', color='cadetblue')

        plt.title(title)
        plt.show()


def scatter_plot_pca(classification, features):
    pca = get_pca(features, 2)
    df = pd.DataFrame(pca, columns=["Comp1", "Comp2"])
    df["Classification"] = classification

    fig, ax = plt.subplots()

    colors = {
        "0": "deeppink",
        "1": "greenyellow",
        "2": "orange",
        "3": "blueviolet",
        "4": "thistle",
        "5": "deepskyblue",
    }

    grouped = df.groupby("Classification")

    for key, group in grouped:
        group.plot(
            ax=ax, kind="scatter", x="Comp1", y="Comp2", label=key, color=colors[key]
        )
    plt.show()


if __name__ == "__main__":
    # classifications = np.array(get_classes(r"D:\proj3_data\project3\train.csv"))
    # # classifications = np.array(get_classes(r"D:\repos\CS529_Project3\train1.csv"))
    # classification = classifications[:, 1]
    # directory = r"D:\proj3_data\project3\trainwav"
    # features1 = get_single_mfcc_features(classifications, directory)
    # features2 = get_processed_mfccs(directory, classifications)
    # plot_pca(classification, features1)
    # plot_pca(classification, features2)

    file_list = get_classes(r"D:\repos\CS529_Project3\examples.csv")
    file_list = np.array(file_list)
    print(file_list.shape)
    train_dir = r"D:\proj3_data\project3\trainwav"
    plot_pca(file_list, train_dir)
