import argparse
import numpy as np

from visualization.mfccs import vizualize_individual_mfccs
from utils.utils import get_classes
from visualization.pca import get_features
from visualization.pca import scatter_plot_pca
from visualization.spectrogram import display_spectrogram

from classification.svm import csv_driver
from classification.svm import driver
from classification.neuralnet import NNModel
from classification.neuralnet import EnsembleNNModel


def visualize():
    """
    runs all visualization and plots
    :return: none
    """
    # MFCCs
    file_list = get_classes("visualization/examples.csv")
    file_arr = np.array(file_list)
    directory = "./visualization/wavs"
    vizualize_individual_mfccs(file_arr, directory)

    # PCA
    features = get_features(file_arr, directory)
    classification = file_arr[:, 1]
    scatter_plot_pca(classification, features)

    # Mel-spectrogram
    display_spectrogram(file_arr, directory)


def build_parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="parse ml args")
    parser.add_argument(
        "-train",
        type=str,
        default="D:/proj3_data/project3/train",
        help="directory containing training mp3s",
    )
    parser.add_argument(
        "-test",
        type=str,
        default="D:/proj3_data/project3/test",
        help="directory containing testing mp3s",
    )
    parser.add_argument(
        "-visualize",
        default=False,
        action="store_true",
        help="pass this to visualize the data",
    )
    parser.add_argument(
        "-svm",
        default=False,
        action="store_true",
        help="run the SVM model using the pre-processed data",
    )
    parser.add_argument(
        "-svm_wav",
        default=False,
        action="store_true",
        help="run the SVM model using wav files. see README for where to put the files",
    )
    parser.add_argument(
        "-neuralnet",
        default=False,
        action="store_true",
        help="run the Neural Network model using the pre-processed data",
    )
    parser.add_argument(
        "-ensemble",
        default=False,
        action="store_true",
        help="run the Ensemble model using the pre-processed data",
    )
    parser.add_argument(
        "-ensemble_count",
        type=int,
        default=10,
        help="number of agents added to the Ensemble",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    options, _ = parser.parse_known_args()

    if options.visualize:
        print("visualizing wav files")
        visualize()
    else:
        print("nothing to see here")


    if options.svm:
        print("running SVM with preprocessed data")
        csv_driver("data.csv", "testdata.csv")

    if options.svm_wav:
        print("running SVM with wav directories")
        driver("train", "test", "train.csv", "test.csv")

    if options.neuralnet:
        print("Running Neural Network with preprocessed data")
        NNModel()

    if options.ensemble:
        print("Running Ensemble with" + str(options.ensemble_count) + "Neural Networks")
        EnsembleNNModel(options.ensemble_count)

