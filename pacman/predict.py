import sys
import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.externals import joblib


# run_regressor_classifier_model_1 is the first thermal model implemented in PACMAN which is 
# T_int[t+1] - T_int[t] = alpha * (T_ext[t] - T_int[t]) + beta * S[t] + gamma * (T_ext[t]/30) + epsilon
# T_ext is divided by 30 to normalise it with other features
# Terminology:
#   alpha = leakage rate
#   beta = cooling rate
#   gamma, epsilon = coefficient of thermal noise
# Input:
#   df_pred - Prediction frame
#   Tset - Set temperature for the duration
#   fregr - learned regression model to predict room temperature
#   fclass - learned classifier model to predict AC compressor state
# Output:
#   df_pred - predicted temperature and AC compressor state for the usage
def run_regressor_classifier_model_1(df_pred, Tset, fregr, fclass):

    # Start and stop time of AC 
    tidx = df_pred[df_pred["status"] == 1].index[0]
    lidx = df_pred[df_pred["status"] == 1].tail(1).index[0]

    # Extract AC usage data
    df_pred = df_pred[tidx:lidx]
    Ticurr = df_pred["int_temperature"].values[0]
    Tscurr = df_pred["status"].values[0]

    # Initialise an empty frame
    df_pred.loc[:, "T_pred"] = 0
    df_pred.loc[:, "S_pred"] = 0

    df_pred.loc[tidx, "T_pred"] = df_pred.loc[tidx, "int_temperature"]
    df_pred.loc[tidx, "S_pred"] = df_pred.loc[tidx, "status"]

    # Learned parameters - Catch file not found error
    try:
        regr = joblib.load(fregr)
        clf = joblib.load(fclass)
    except Exception as error:
        print error

    # Feature set
    v = Ticurr
    text = df_pred["ext_temperature"].values[0]
    p1 = text - v
    p2 = Tscurr
    p3 = numpy.divide(text, 30)

    # Predict for whole duration
    for idx in df_pred.index:

        # Predict temperature difference
        diff = regr.predict([[p1, p2, p3]])

        # Predict temperature
        df_pred.loc[idx+1, "T_pred"] = v + diff[0]

        # Update feature vector
        v = df_pred.loc[idx+1, "T_pred"]
        text = df_pred.loc[idx+1, "ext_temperature"]

        p1 = text - v
        p2 = clf.predict([[v - Tset, diff]])[0]
        p3 = numpy.divide(text, 30)

        # Predicted AC compressor state
        df_pred.loc[idx+1, "S_pred"] = p2

    # Return predicted frame
    return df_pred[:-1]


# run_regressor_classifier_model_2 is the second thermal model implemented in PACMAN which is 
# T_int[t+1] - T_int[t] = alpha * (T_ext[t] - T_int[t]) + beta * S[t] + epsilon
# Terminology:
#   alpha = leakage rate
#   beta = cooling rate
#   epsilon = coefficient of thermal noise
# Input:
#   df_pred - Prediction frame
#   Tset - Set temperature for the duration
#   fregr - learned regression model to predict room temperature
#   fclass - learned classifier model to predict AC compressor state
# Output:
#   df_pred - predicted temperature and AC compressor state for the usage
def run_regressor_classifier_model_2(df_pred, Tset, fregr, fclass):

    # Start and stop time of AC 
    tidx = df_pred[df_pred["status"] == 1].index[0]
    lidx = df_pred[df_pred["status"] == 1].tail(1).index[0]

    # Extract AC usage data
    df_pred = df_pred[tidx:lidx]
    Ticurr = df_pred["int_temperature"].values[0]
    Tscurr = df_pred["status"].values[0]

    # Initialise an empty frame
    df_pred.loc[:, "T_pred"] = 0
    df_pred.loc[:, "S_pred"] = 0

    df_pred.loc[tidx, "T_pred"] = df_pred.loc[tidx, "int_temperature"]
    df_pred.loc[tidx, "S_pred"] = df_pred.loc[tidx, "status"]

    # Learned parameters - Catch file not found error
    try:
        regr = joblib.load(fregr)
        clf = joblib.load(fclass)
    except Exception as error:
        print error

    # Feature set
    v = Ticurr
    text = df_pred["ext_temperature"].values[0]
    p1 = text - v
    p2 = Tscurr

    # Predict for whole duration
    for idx in df_pred.index:

        # Predict temperature difference
        diff = regr.predict([[p1, p2]])

        # Predict temperature
        df_pred.loc[idx+1, "T_pred"] = v + diff[0]

        # Update feature vector
        v = df_pred.loc[idx+1, "T_pred"]
        text = df_pred.loc[idx+1, "ext_temperature"]

        p1 = text - v
        p2 = clf.predict([[v - Tset, diff]])[0]

        # Predicted AC compressor state
        df_pred.loc[idx+1, "S_pred"] = p2

    # Return predicted frame
    return df_pred[:-1]


# run_regressor_classifier_model_3 is the third thermal model implemented in PACMAN which is 
# T_int[t+1] - T_int[t] = alpha * (T_int[t]/25) + beta * S[t] + gamma * (T_ext[t]/25) + epsilon
# Terminology:
#   alpha = leakage rate
#   beta = cooling rate
#   gamma, epsilon = coefficient of thermal noise
# Input:
#   df_pred - Prediction frame
#   Tset - Set temperature for the duration
#   fregr - learned regression model to predict room temperature
#   fclass - learned classifier model to predict AC compressor state
# Output:
#   df_pred - predicted temperature and AC compressor state for the usage
def run_regressor_classifier_model_3(df_pred, Tset, fregr, fclass):

    # Start and stop time of AC 
    tidx = df_pred[df_pred["status"] == 1].index[0]
    lidx = df_pred[df_pred["status"] == 1].tail(1).index[0]

    # Extract AC usage data
    df_pred = df_pred[tidx:lidx]
    Ticurr = df_pred["int_temperature"].values[0]
    Tscurr = df_pred["status"].values[0]

    # Initialise an empty frame
    df_pred.loc[:, "T_pred"] = 0
    df_pred.loc[:, "S_pred"] = 0

    df_pred.loc[tidx, "T_pred"] = df_pred.loc[tidx, "int_temperature"]
    df_pred.loc[tidx, "S_pred"] = df_pred.loc[tidx, "status"]

    # Learned parameters - Catch file not found error
    try:
        regr = joblib.load(fregr)
        clf = joblib.load(fclass)
    except Exception as error:
        print error

    # Feature set
    v = Ticurr
    text = df_pred["ext_temperature"].values[0]
    p1 = v/25
    p2 = Tscurr
    p3 = text/25

    # Predict for whole duration
    for idx in df_pred.index:

        # Predict temperature difference
        diff = regr.predict([[p1, p2, p3]])

        # Predict temperature
        df_pred.loc[idx+1, "T_pred"] = v + diff[0]

        # Update feature vector
        v = df_pred.loc[idx+1, "T_pred"]
        text = df_pred.loc[idx+1, "ext_temperature"]

        p1 = v/25
        p2 = clf.predict([[v - Tset, diff[0]]])[0]
        p3 = text/25

        # Predicted AC compressor state
        df_pred.loc[idx+1, "S_pred"] = p2

    # Return predicted frame
    return df_pred[:-1]


# run_regressor_classifier_model_4 is the fourth thermal model implemented in PACMAN which is 
# T_int[t+1] = alpha * T_int[t] + beta * S[t] + epsilon
# Terminology:
#   alpha = leakage rate
#   beta = cooling rate
#   epsilon = coefficient of thermal noise
# Input:
#   df_pred - Prediction frame
#   Tset - Set temperature for the duration
#   fregr - learned regression model to predict room temperature
#   fclass - learned classifier model to predict AC compressor state
# Output:
#   df_pred - predicted temperature and AC compressor state for the usage
def run_regressor_classifier_model_4(df_pred, Tset, fregr, fclass):

    # Start and stop time of AC 
    tidx = df_pred[df_pred["status"] == 1].index[0]
    lidx = df_pred[df_pred["status"] == 1].tail(1).index[0]

    # Extract AC usage data
    df_pred = df_pred[tidx:lidx]
    Ticurr = df_pred["int_temperature"].values[0]
    Tscurr = df_pred["status"].values[0]

    # Initialise an empty frame
    df_pred.loc[:, "T_pred"] = 0
    df_pred.loc[:, "S_pred"] = 0

    df_pred.loc[tidx, "T_pred"] = df_pred.loc[tidx, "int_temperature"]
    df_pred.loc[tidx, "S_pred"] = df_pred.loc[tidx, "status"]

    # Learned parameters - Catch file not found error
    try:
        regr = joblib.load(fregr)
        clf = joblib.load(fclass)
    except Exception as error:
        print error

    # Feature set
    v = Ticurr
    text = df_pred["ext_temperature"].values[0]
    p1 = v
    p2 = Tscurr

    # Predict for whole duration
    for idx in df_pred.index:

        # Predict temperature difference
        v = regr.predict([[p1, p2]])[0]

        # Predict temperature
        df_pred.loc[idx+1, "T_pred"] = v

        # Update feature vector
        diff = v - df_pred.loc[idx, "T_pred"]

        p1 = v
        p2 = clf.predict([[v - Tset, diff]])[0]

        # Predicted AC compressor state
        df_pred.loc[idx+1, "S_pred"] = p2

    # Return predicted frame
    return df_pred[:-1]


# run_regressor_classifier_model_5 uses the same thermal model as the fourth thermal model but 
# uses threshold based state prediction instead of a learned classifier. Following is the implemented 
# thermal model for the scenario.
# T_int[t+1] = alpha * T_int[t] + beta * S[t] + epsilon
# Terminology:
#   alpha = leakage rate
#   beta = cooling rate
#   epsilon = coefficient of thermal noise
# Input:
#   df_pred - Prediction frame
#   Tset - Set temperature for the duration
#   fregr - learned regression model to predict room temperature
# Output:
#   df_pred - predicted temperature and AC compressor state for the usage
def run_regressor_classifier_model_5(df_pred, Tset, fregr):

    # Start and stop time of AC 
    tidx = df_pred[df_pred["status"] == 1].index[0]
    lidx = df_pred[df_pred["status"] == 1].tail(1).index[0]

    # Extract AC usage data
    df_pred = df_pred[tidx:lidx]
    Ticurr = df_pred["int_temperature"].values[0]
    Tscurr = df_pred["status"].values[0]

    # Initialise an empty frame
    df_pred.loc[:, "T_pred"] = 0
    df_pred.loc[:, "S_pred"] = 0

    df_pred.loc[tidx, "T_pred"] = df_pred.loc[tidx, "int_temperature"]
    df_pred.loc[tidx, "S_pred"] = df_pred.loc[tidx, "status"]

    # Learned parameters - Catch file not found error
    try:
        regr = joblib.load(fregr)
    except Exception as error:
        print error

    # Feature set
    v = Ticurr
    p1 = v
    p2 = Tscurr

    # Fixed on and off hysteresis
    T_on = Tset + 2
    T_off = Tset

    # Predict for whole duration
    for idx in df_pred.index:

        # Predict temperature difference
        v = regr.predict([[p1, p2]])[0]

        # Predict temperature
        df_pred.loc[idx+1, "T_pred"] = v

        # Update feature vector
        diff = v - df_pred.loc[idx, "T_pred"]

        # Threshold based AC compressor state prediction
        p1 = v
        if v <= df_pred.loc[idx, "T_pred"]:
            if v > T_off:
                p2 = 1
            elif v <= T_off:
                p2 = 0

        elif v > df_pred.loc[idx, "T_pred"]:
            if v <= T_on:
                p2 = 0
            elif v > T_on:
                p2 = 1

        # Predicted AC compressor state
        df_pred.loc[idx+1, "S_pred"] = p2

    # Return predicted frame
    return df_pred[:-1]