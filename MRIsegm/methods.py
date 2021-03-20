import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
from itertools import product
import cv2

from MRIsegm.processing import gabor_filter

__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


def features_extraction(slice, layer, ksize, sigma, theta, lamb, gamma, psi):

    df = pd.DataFrame()  # create dataframe for features

    img = slice[layer, :, :].copy()

    img_vect = img.reshape(-1)  # reshape 2D image into 1D vector
    df['Original pixels'] = img_vect

    # generate a bunch of Gabor filters

    kernels_list = list(
        map(gabor_filter, *zip(*product(ksize, sigma, theta, lamb, gamma, psi))))

    for i, kernel in enumerate(kernels_list):
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        fimg_vect = fimg.reshape(-1)

        Gabor_label = "Gabor_" + str(i)
        df[Gabor_label] = fimg_vect

    return df


def definig_variables(df, test_size=0.35, random_state=20):

    Y = df['labels'].values
    X = df.drop(labels=['labels'], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)

    return X_train, X_test, Y_train, Y_test


def print_metrics(Y_test, prediction_test):
    print("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))

    # deal with imbalanced datasets.
    print("Balanced accuracy = ", metrics.balanced_accuracy_score(
        Y_test, prediction_test))

    # macro: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    print("f1_score (macro) = ", metrics.f1_score(
        Y_test, prediction_test, average='macro'))

    # Calculate metrics globally by counting the total true positives, false negatives and false positives
    print("f1_score (micro) = ", metrics.f1_score(
        Y_test, prediction_test, average='micro'))
    metrics.homogeneity_score
    print("homogeneity_score = ", metrics.homogeneity_score(
        Y_test, prediction_test))


def print_features_importances(X, model):

    features_list = list(X.columns)
    features_imp = pd.Series(model.feature_importances_,
                             index=features_list).sort_values(ascending=False)

    pd.options.display.float_format = '{:,.10f}'.format
    print(features_imp)


def save_model(filename, model):

    if type(filename) != str:
        filename = str(filename)
    pickle.dump(model, open(filename, 'wb'))  # wb - write binary


def load_model(filename):

    loaded_model = pickle.load(open(filename, 'rb'))  # rb - read binary

    return loaded_model


def show_model_result(ground_truth, predicted):

    fig, ax = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)

    ax[0].imshow(predicted, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("predicted")

    ax[1].imshow(ground_truth, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("ground truth")
