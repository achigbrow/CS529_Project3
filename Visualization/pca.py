from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from features import get_classes
from features import get_mfccs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_pca(features):
    # nsamples, nfeatures = features.shape
    # n = min(nsamples, nfeatures)
    pca = PCA(n_components=2)
    transformed = pca.fit(features).transform(features)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    return scaler.transform(transformed)


if __name__ == "__main__":
    data = get_classes(r"D:\proj3_data\project3\train.csv")
    # data = get_classes(r"D:\repos\CS529_Project3\train1.csv")
    mfcc = get_mfccs(r"D:\proj3_data\project3\trainwav", data)
    # get rid of the last col that holds classification
    features = mfcc[:, :-1]
    classification = mfcc[:, -1]
    pca_mfcc = get_pca(features)

    df = pd.DataFrame(pca_mfcc, columns=["Comp1", "Comp2"])
    df["Classification"] = classification
    unique_classes = df["Classification"].unique

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
