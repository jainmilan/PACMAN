__author__ = 'milan'

import os
import sys
import numpy
import pandas
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


def run_classifier_model_1(df, usage):
    data_size = len(df)

    if data_size < 30:
        return None

    df["idiff"] = df.int_temperature.diff()
    df["isdiff"] = df.int_temperature - df.Tset

    feature_matrix_classify = pandas.DataFrame()
    feature_matrix_classify["isdiff"] = df.tail(data_size-1).reset_index()["isdiff"]
    feature_matrix_classify["idiff"] = df.tail(data_size-1).reset_index()["idiff"]

    Y_classify = df.tail(data_size-1).reset_index().status

    classify = KNeighborsClassifier()
    classify.fit(feature_matrix_classify, Y_classify)

    model_dir = '../Results/Prediction/models/clf/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(classify, model_dir + 'knn.pkl')
    return True


def run_regr_model_1(df):
    data_size = len(df)

    # Internal and External Temperature
    Tint = df['int_temperature'].values
    Text = df['ext_temperature'].values
    Text_norm = numpy.divide(Text, 30)
    status = df['status'].values

    if data_size < 30:
        return None

    # Parameters for our model
    diff_int_temp = numpy.ediff1d(Tint)

    diff_temperature = Text - Tint

    # Matrix for learning the Model
    m = data_size - 1
    feature_matrix = numpy.zeros((m, 3))
    feature_matrix[:, 0] = diff_temperature[0:-1]
    feature_matrix[:, 1] = status[0:-1]
    feature_matrix[:, 2] = Text_norm[0:-1]

    Y_linear = diff_int_temp

    regr = KNeighborsRegressor()
    regr.fit(feature_matrix, Y_linear)

    model_dir = '../Results/Prediction/models/regr/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(regr, model_dir + 'knn_1.pkl')
    return True


def run_regr_model_2(df):
    data_size = len(df)

    # Internal and External Temperature
    Tint = df['int_temperature'].values
    Text = df['ext_temperature'].values
    status = df['status'].values

    if data_size < 30:
        return None

    # Parameters for our model
    diff_int_temp = numpy.ediff1d(Tint)

    diff_temperature = Text - Tint

    # Matrix for learning the Model
    m = data_size - 1
    feature_matrix = numpy.zeros((m, 2))
    feature_matrix[:, 0] = diff_temperature[0:-1]
    feature_matrix[:, 1] = status[0:-1]

    Y_linear = diff_int_temp

    regr = KNeighborsRegressor()
    regr.fit(feature_matrix, Y_linear)

    model_dir = '../Results/Prediction/models/regr/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(regr, model_dir + 'knn_2.pkl')
    return True

