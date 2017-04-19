__author__ = 'milan'

import os
from sklearn.externals import joblib

# Models
from sklearn import linear_model
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def run_classifier(feature_matrix, Y, classifier, class_type, class_dir):
    # Initialization
    classify = None

    if classifier == "lr":
        classify = linear_model.LogisticRegression()
    elif classifier == "svm":
        classify = LinearSVC()
    elif classifier == "rf":
        classify = RandomForestClassifier(random_state=0)
    elif classifier == "knn":
        classify = KNeighborsClassifier()
    elif classifier == "dt":
        classify = DecisionTreeClassifier(random_state=0)

    # Invalid Model
    try:
        classify.fit(feature_matrix, Y)
    except Exception as error:
        print error

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    joblib.dump(classify, class_dir + classifier + '_' + str(class_type) + '.pkl')
    return True


def run_regressor(feature_matrix, Y, regressor, regress_type, regress_dir):
    # Initialization
    regr = None

    if regressor == "lr":
        regr = linear_model.LinearRegression()
    elif regressor == "svm":
        regr = LinearSVR()
    elif regressor == "rf":
        regr = RandomForestRegressor(random_state=0)
    elif regressor == "knn":
        regr = KNeighborsRegressor()
    elif regressor == "dt":
        regr = DecisionTreeRegressor(random_state=0)

    # Invalid Model
    try:
        regr.fit(feature_matrix, Y)
    except Exception as error:
        print error

    if not os.path.exists(regress_dir):
        os.makedirs(regress_dir)

    joblib.dump(regr, regress_dir + regressor + '_' + str(regress_type) + '.pkl')

    return True
