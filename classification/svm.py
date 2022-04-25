from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from utils.utils import write_submission
from utils.utils import get_classes
from visualization.mfccs import get_processed_mfccs
import numpy as np
import matplotlib as plt


def classify(X, y, Z):
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5)).fit(X, y)
    plot_support_vectors(X, clf)
    prediction = clf.predict(Z)
    return prediction

def predict_classes(Z, clf):


def plot_support_vectors(X, clf):
  plt.figure(figsize=(10, 5))
  for i, C in enumerate([1, 100]):
    decision_function = clf.decision_function(X)
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15[0])
    support_vectors = X[support_vector_indices]

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
      rand_train_genres[index] = train_classifications[index][1]

    test_ids = np.array(get_classes(test_csv))
    train_mfccs = get_processed_mfccs(train_dir, train_classifications)
