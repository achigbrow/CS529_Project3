from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from utils import get_classes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mfccs import get_single_mfcc_features
from mfccs import get_processed_mfccs


def get_pca(features):

    pca = PCA(n_components=2)
    transformed = pca.fit(features).transform(features)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    return scaler.transform(transformed)


def plot_pca(classification, features):
    pca = get_pca(features)
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
    classifications = np.array(get_classes(r"D:\proj3_data\project3\train.csv"))
    # classifications = np.array(get_classes(r"D:\repos\CS529_Project3\train1.csv"))
    classification = classifications[:, 1]
    directory = r"D:\proj3_data\project3\trainwav"
    features1 = get_single_mfcc_features(classifications, directory)
    features2 = get_processed_mfccs(directory, classifications)
    plot_pca(classification, features1)
    plot_pca(classification, features2)
