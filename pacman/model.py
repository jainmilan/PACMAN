__author__ = 'milan'

# Inbuilt libraries
import os
from sklearn.externals import joblib

# Inbuild machine learning models
from sklearn import linear_model
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# run_classifier function combines all the historical usages and generates a common frame
# to learn a regressor and a classifier
# Input:
#   feature_matrix - matrix having the feature set
#   Y - output matrix
#   classifier - machine learning algorithm to use as classifier
#   class_type - type of class - classifier based or threshold driven
#   class_dir - address to save the learned classifier
# Output:
#   True - If parameters are properly learned
def run_classifier(feature_matrix, Y, classifier, class_type, class_dir):
    
    # Initialization
    classify = None

    # Select a classifier
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

    # Create the path if doesn't exist and save the model parameters
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    joblib.dump(classify, class_dir + classifier + '_' + str(class_type) + '.pkl')

    # Return success
    return True


# run_regressor function combines all the historical usages and generates a common frame
# to learn a regressor and a classifier
# Input:
#   feature_matrix - matrix having the feature set
#   Y - output matrix
#   regressor - machine learning algorithm to use as regressor
#   regress_type - type of class - regressor based or threshold driven
#   regress_dir - address to save the learned regressor
# Output:
#   True - If parameters are properly learned
def run_regressor(feature_matrix, Y, regressor, regress_type, regress_dir):

    # Initialization
    regr = None

    # Select a regressor
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

    # Create the path if doesn't exist and save the model parameters
    if not os.path.exists(regress_dir):
        os.makedirs(regress_dir)

    joblib.dump(regr, regress_dir + regressor + '_' + str(regress_type) + '.pkl')

    # Return success
    return True
