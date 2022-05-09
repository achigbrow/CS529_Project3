from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.utils import write_submission
from utils.utils import get_classes

from visualization.mfccs import get_processed_mfccs
import numpy as np
import tensorflow as tf
import pandas as pd
import time
from sklearn.svm import SVC


def classify(X, y, X_test, y_test, Z):
    """Classifies Z using a Linear SVM, prints 80/20 split and creates confusion matrix

    :param X: training features
    :param y: training labels
    :param X_test: testing features
    :param y_test: testing labels
    :param Z: features to classify
    :return: prediction of Z classes
    """
    # TODO: experiment with values of C and gamma
    # uses rbf kernel
    clf = make_pipeline(StandardScaler(), SVC(gamma="auto", C=5)).fit(X, y)
    # get predictions
    submission = clf.predict(Z)

    # Obtain the training and testing accuracy scores
    train = clf.score(X, y)
    test = clf.score(X_test, y_test)
    # Get test predictions for confusion matrix
    predicted = clf.predict(X_test)
    # Convert everything to int to avoid errors with confusion matrix
    predicted = predicted.astype(int)
    y_test = y_test.astype(int)

    # display accuracies for record
    print("train accuracy", train)
    print("test accuracy", test)

    # build confusion matrix
    conf_mat = tf.math.confusion_matrix(
        labels=y_test, predictions=predicted, num_classes=6
    ).numpy()
    np.savetxt("SVMconfusion.csv", conf_mat, delimiter=",")

    return submission


def driver(train_dir, test_dir, train_csv, test_csv):
    """prepares the data, calls the classifying model, and returns submission file
        for use with wav files

    :param train_dir: directory with training data in wav form
    :param test_dir:  directory with testing data in wav form
    :param train_csv: filepath to csv containing training information
    :param test_csv:  filepath to csv containing ids of files to predict
    :return: none
    """
    # get the training data
    train_classifications = np.array(get_classes(train_csv))
    train_ids = train_classifications[:, 0]
    train_genres = train_classifications[:, 1]

    # Split once to get the test and training set
    X_train, X_test, y_train, y_test = train_test_split(
        train_ids, train_genres, test_size=0.2, random_state=123, stratify=train_genres
    )

    scaler = StandardScaler()

    # get the data to make predictions about
    submission_ids = get_classes(test_csv)
    submission_ids_new = [id for sublist in submission_ids for id in sublist]

    # obtain processed mfccs as training features for the model
    train_s = time.time()
    train_features = get_processed_mfccs(train_dir, X_train)
    train_features = scaler.fit_transform(train_features)
    train_e = time.time()
    print("train mfcc", train_e - train_s)

    # obtain processed mfccs as testing features for the model
    test_s = time.time()
    test_features = get_processed_mfccs(train_dir, X_test)
    test_features = scaler.fit_transform(test_features)
    test_e = time.time()
    print("test mfcc", test_e - test_s)

    # obtain features for the files to classify
    submit_s = time.time()
    submit_features = get_processed_mfccs(test_dir, submission_ids_new)
    submit_features = scaler.fit_transform(submit_features)
    submit_e = time.time()
    print("test mfcc", submit_e - submit_s)

    # genres of the associated features
    train_labels = y_train
    test_labels = y_test

    # make predictions using Linear SVM
    predict_s = time.time()
    predictions = classify(
        train_features, train_labels, test_features, test_labels, submit_features
    )
    predictions = predictions.astype(int)
    predict_e = time.time()
    print("prediction", predict_e - predict_s)

    # create submission csv
    write_submission(submission_ids_new, predictions)


def csv_driver(data, testdata):
    """uses a csv of prepared features

    prints a predictions.csv for submission

    :param data: filepath to csv containing training features
    :param testdata: filepath to csv containing test features
    :return: none
    """
    data = pd.read_csv(data)

    # Drop unnecessary column
    data = data.drop(["filename"], axis=1)

    genre_list = data.iloc[:, -1]
    y = genre_list

    # normalizing
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Predict the test data
    testdata = pd.read_csv(testdata)

    # Drop unnecessary column
    testdata_raw = testdata.drop(["filename"], axis=1)

    # normalizing
    scaler = StandardScaler()
    Z = scaler.fit_transform(np.array(testdata_raw.iloc[:, :-1], dtype=float))

    test_predictions = classify(X_train, y_train, X_test, y_test, Z)

    prediction_df = pd.DataFrame(zip(testdata.filename, test_predictions))

    prediction_df.rename(columns={0: "id", 1: "genre"}, inplace=True)

    prediction_df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    start = time.time()
    csv_driver("../data.csv", "../testdata.csv")
    end = time.time()
    print(end - start)
