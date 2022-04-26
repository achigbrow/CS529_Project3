from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import write_submission
from utils import get_classes

from visualization.mfccs import get_processed_mfccs
import numpy as np
import matplotlib.pyplot as plt
import time


def classify(X, y, Z):
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5)).fit(X, y)
    # plot_support_vectors(X, y, clf)
    prediction = clf.predict(Z)
    return prediction


def plot_support_vectors(X, y, clf):
    """needs debugging"""
    #TODO
    plt.figure(figsize=(10, 5))
    for i, C in enumerate([1, 100]):
        decision_function = clf.decision_function(X)
        support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
        support_vectors = X[support_vector_indices]
        y = y.astype(int)

        plt.subplot(1, 2, i + 1)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx, yy = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50)
        )
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(
            xx,
            yy,
            Z,
            colors="k",
            levels=[-1, 0, 1],
            alpha=0.5,
            linestyles=["--", "-", "--"],
        )
        plt.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            s=100,
            linewidth=1,
            facecolors="none",
            edgecolors="k",
        )
        plt.title("C=" + str(C))
        plt.tight_layout()
        plt.show()

def driver(train_dir, test_dir, train_csv, test_csv):
    train_classifications = np.array(get_classes(train_csv))
    train_ids = train_classifications[:, 0]
    train_genres = train_classifications[:, 1]
    rand_train_ids = np.random.choice(train_ids, 1600)
    rand_train_genres = np.zeros(len(rand_train_ids))

    for i in range(len(rand_train_ids)):
        index = np.where(train_ids == rand_train_ids[i])
        rand_train_genres[i] = train_classifications[index, 1]


    test_ids = get_classes(test_csv)
    test_ids_new = [id for sublist in test_ids for id in sublist]

    train_s = time.time()
    train_mfccs = get_processed_mfccs(train_dir, rand_train_ids)
    train_e = time.time()
    print('train mfcc', train_e - train_s)
    test_s = time.time()
    test_mfccs = get_processed_mfccs(test_dir, list(test_ids_new))
    test_e = time.time()
    print('test mfcc', test_e - test_s)
    predict_s = time.time()
    predictions = classify(train_mfccs, train_genres, test_mfccs)
    predict_e = time.time()
    print('prediction', predict_e - predict_s)
    submit_ids = get_classes(test_csv, False)
    write_submission(submit_ids, predictions)

if __name__ == "__main__":
    # classifications = get_classes(r"D:\proj3_data\project3\train.csv")
    # classifications = get_classes(r"D:\repos\CS529_Project3\train1.csv")
    train_dir = r"D:\proj3_data\project3\trainwav"
    # train_csv = r"D:\repos\CS529_Project3\train1.csv"
    train_csv = r"D:\proj3_data\project3\train.csv"
    # test_csv = r"D:\repos\CS529_Project3\test1.csv"
    test_csv = r"D:\proj3_data\project3\test_idx.csv"
    test_dir = r"D:\proj3_data\project3\testwav"
    start = time.time()
    driver(train_dir, test_dir, train_csv, test_csv)
    end = time.time()
    print(end - start)
