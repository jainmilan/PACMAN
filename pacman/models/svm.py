import os
import numpy
import pandas

from sklearn.externals import joblib
from sklearn.svm import SVR, LinearSVR, LinearSVC


def run(df):
    data_size = len(df)
    Tset = 0

    # Internal and External Temperature
    Tint = df['int_temperature'].values
    Text = df['ext_temperature'].values
    status = df['status'].values

    Ticurr = Tint[0]
    Tilast = Tint[-1]
    Tistd = numpy.std(Tint)
    Tivar = numpy.var(Tint)
    Tidiff = Tilast - Ticurr

    Tecurr = Text[0]
    Telast = Text[-1]
    Testd = numpy.std(Text)
    Tevar = numpy.var(Text)
    Tediff = Telast - Tecurr

    Tie = numpy.subtract(Text, Tint)
    Tiecurr = Tie[0]
    Tielast = Tie[-1]
    Tiestd = numpy.std(Tie)
    Tievar = numpy.var(Tie)
    Tiediff = Tielast - Tiecurr

    ocurr = numpy.sum(status)

    if data_size < 30:
        return pandas.Series({'alpha':0, 'beta':0, 'epsilon':0, 'Text':numpy.mean(Text), 'V':0, 'size':data_size,
                              'Ticurr':Ticurr, 'Tilast':Tilast, 'Tistd':Tistd, 'Tivar':Tivar, 'Tidiff': Tidiff,
                              'Tecurr':Tecurr, 'Telast':Telast, 'Testd':Testd, 'Tevar':Tevar, 'Tediff': Tediff,
                              'Tiecurr':Tiecurr, 'Tielast':Tielast, 'Tiestd':Tiestd, 'Tievar':Tievar, 'Tiediff': Tiediff,
                              'ocurr': ocurr})

    # Parameters for our model
    diff_int_temp = numpy.ediff1d(Tint)

    diff_temperature = Text - Tint

    # Matrix for learning the Model
    m = data_size - 1
    feature_matrix = numpy.zeros((m, 2))
    feature_matrix[:, 0] = diff_temperature[0:-1]
    feature_matrix[:, 1] = status[0:-1]

    Y_linear = diff_int_temp

    regr = SVR(kernel='linear')
    regr.fit(feature_matrix, Y_linear)

    alpha = regr.coef_[0][0]*-1.0
    beta = regr.coef_[0][1]*-1.0
    epsilon = regr.intercept_[0]*-1.0

    return pandas.Series({'alpha':alpha, 'beta':beta, 'epsilon':epsilon, 'Text':numpy.mean(Text), 'V':1, 'size':data_size,
                              'Ticurr':Ticurr, 'Tilast':Tilast, 'Tistd':Tistd, 'Tivar':Tivar, 'Tidiff': Tidiff,
                              'Tecurr':Tecurr, 'Telast':Telast, 'Testd':Testd, 'Tevar':Tevar, 'Tediff': Tediff,
                              'Tiecurr':Tiecurr, 'Tielast':Tielast, 'Tiestd':Tiestd, 'Tievar':Tievar, 'Tiediff': Tiediff,
                              'ocurr': ocurr})


def run_2(df):
    data_size = len(df)
    Tset = 0

    # Internal and External Temperature
    Tint = df['int_temperature'].values
    Text = df['ext_temperature'].values
    status = df['status'].values

    Ticurr = Tint[0]
    Tilast = Tint[-1]
    Tistd = numpy.std(Tint)
    Tivar = numpy.var(Tint)
    Tidiff = Tilast - Ticurr

    Tecurr = Text[0]
    Telast = Text[-1]
    Testd = numpy.std(Text)
    Tevar = numpy.var(Text)
    Tediff = Telast - Tecurr

    Tie = numpy.subtract(Text, Tint)
    Tiecurr = Tie[0]
    Tielast = Tie[-1]
    Tiestd = numpy.std(Tie)
    Tievar = numpy.var(Tie)
    Tiediff = Tielast - Tiecurr

    ocurr = numpy.sum(status)

    if data_size < 30:
        return pandas.Series({'alpha':0, 'beta':0, 'epsilon':0, 'Text':numpy.mean(Text), 'V':0, 'size':data_size,
                              'Ticurr':Ticurr, 'Tilast':Tilast, 'Tistd':Tistd, 'Tivar':Tivar, 'Tidiff': Tidiff,
                              'Tecurr':Tecurr, 'Telast':Telast, 'Testd':Testd, 'Tevar':Tevar, 'Tediff': Tediff,
                              'Tiecurr':Tiecurr, 'Tielast':Tielast, 'Tiestd':Tiestd, 'Tievar':Tievar, 'Tiediff': Tiediff,
                              'ocurr': ocurr})

    # Parameters for our model
    diff_int_temp = numpy.ediff1d(Tint)

    diff_temperature = Text - Tint

    # Matrix for learning the Model
    m = data_size - 1
    feature_matrix = numpy.zeros((m, 2))
    feature_matrix[:, 0] = Text[0:-1]
    feature_matrix[:, 1] = status[0:-1]

    Y_linear = diff_int_temp

    #regr = SVR(kernel='linear')
    regr = LinearSVR(fit_intercept=False)
    regr.fit(feature_matrix, Y_linear)

    print regr.coef_
    alpha = regr.coef_[0]
    beta = regr.coef_[1]
    #epsilon = regr.intercept_[0]*-1.0
    epsilon = 0

    return pandas.Series({'alpha':alpha, 'beta':beta, 'epsilon':epsilon, 'Text':numpy.mean(Text), 'V':1, 'size':data_size,
                              'Ticurr':Ticurr, 'Tilast':Tilast, 'Tistd':Tistd, 'Tivar':Tivar, 'Tidiff': Tidiff,
                              'Tecurr':Tecurr, 'Telast':Telast, 'Testd':Testd, 'Tevar':Tevar, 'Tediff': Tediff,
                              'Tiecurr':Tiecurr, 'Tielast':Tielast, 'Tiestd':Tiestd, 'Tievar':Tievar, 'Tiediff': Tiediff,
                              'ocurr': ocurr})


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

    classify = LinearSVC()
    classify.fit(feature_matrix_classify, Y_classify)

    model_dir = '../Results/Prediction/models/clf/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(classify, model_dir + 'svm.pkl')
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

    regr = LinearSVR()
    regr.fit(feature_matrix, Y_linear)

    model_dir = '../Results/Prediction/models/regr/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(regr, model_dir + 'svm_1.pkl')
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

    regr = LinearSVR()
    regr.fit(feature_matrix, Y_linear)

    model_dir = '../Results/Prediction/models/regr/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(regr, model_dir + 'svm_2.pkl')
    return True