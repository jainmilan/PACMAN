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
        diff = regr.predict([p1, p2, p3])

        # Predict temperature
        df_pred.loc[idx+1, "T_pred"] = v + diff[0]

        # Update feature vector
        v = df_pred.loc[idx+1, "T_pred"]
        text = df_pred.loc[idx+1, "ext_temperature"]

        p1 = text - v
        p2 = clf.predict([v - Tset, diff])[0]
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
        diff = regr.predict([p1, p2])

        # Predict temperature
        df_pred.loc[idx+1, "T_pred"] = v + diff[0]

        # Update feature vector
        v = df_pred.loc[idx+1, "T_pred"]
        text = df_pred.loc[idx+1, "ext_temperature"]

        p1 = text - v
        p2 = clf.predict([v - Tset, diff])[0]

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
        diff = regr.predict([p1, p2, p3])

        # Predict temperature
        df_pred.loc[idx+1, "T_pred"] = v + diff[0]

        # Update feature vector
        v = df_pred.loc[idx+1, "T_pred"]
        text = df_pred.loc[idx+1, "ext_temperature"]

        p1 = v/25
        p2 = clf.predict([v - Tset, diff[0]])[0]
        p3 = text/25

        # Predicted AC compressor state
        df_pred.loc[idx+1, "S_pred"] = p2

    # Return predicted frame
    return df_pred[:-1]


# run_regressor_classifier_model_4 is the fourth thermal model implemented in PACMAN which is 
# T_int[t+1] - T_int[t] = alpha * T_int[t] + beta * S[t] + epsilon
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
        v = regr.predict([p1, p2])[0]

        # Predict temperature
        df_pred.loc[idx+1, "T_pred"] = v

        # Update feature vector
        diff = v - df_pred.loc[idx, "T_pred"]

        p1 = v
        p2 = clf.predict([v - Tset, diff])[0]

        # Predicted AC compressor state
        df_pred.loc[idx+1, "S_pred"] = p2

    # Return predicted frame
    return df_pred[:-1]


# run_regressor_classifier_model_5 uses the same thermal model as the fourth thermal model but 
# uses threshold based state prediction instead of a learned classifier. Following is the implemented 
# thermal model for the scenario.
# T_int[t+1] - T_int[t] = alpha * T_int[t] + beta * S[t] + epsilon
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
        v = regr.predict([p1, p2])[0]

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

#<<-------------------------------------------------- Deprecated ----------------------------------------------->>#

def euler_dist(data, arg):
    data['Ticurr'] = numpy.square(data['Ticurr'] - arg[0])
    data['Text'] = numpy.square(data['Text'] - arg[1])

    d = numpy.sqrt(data['Ticurr'] + data['Text'])
    return d


def get_parameter(parameters, Tset, T0, Text):
    pValue = 0
    if T0 < numpy.mean(Text):
        pValue = 1

    Tl = int(Tset - 2)
    Tu = int(Tset + 2)

    # param = parameters[(parameters["Tset"].isin(range(Tl, Tu)))]
    param = parameters[(parameters["Phase"] == pValue) & (parameters["Tset"].isin(range(Tl, Tu)))]
    # param = parameters[(parameters["Phase"]==pValue) & (parameters["Tset"]==Tset)]

    if param.empty:
        return pandas.DataFrame(), -1

    param = param.sort(['Score'], ascending=False)[:5]

    euler_param = param[['Tcurrint', 'Tcurrext']].apply(euler_dist, \
                                                        axis=1, arg=(T0, Text[0]))
    min_index = numpy.argmin(euler_param)

    return param, min_index


def get_params(param_file, Tset, hour, room, usage, Ti0, Te0):
    params = pandas.read_csv(param_file, index_col=["ID"])

    room_params = params[params["uid"].str.contains(str(room))]
    if room_params.empty:
        return None
    valid_params = room_params[room_params["V"] == 1]
    if valid_params.empty:
        return None

    hidx = (numpy.abs(valid_params["hour"].values - hour)).argmin()
    vhour = valid_params["hour"].values[hidx]

#    sidx = (numpy.abs(valid_params["Tset"].values - Tset)).argmin()
#    vtset = valid_params["Tset"].values[sidx]

    hour_params = valid_params[valid_params["hour"] == vhour]#[valid_params["Tset"] == vtset]
    if hour_params.empty:
        return None

    euler_param = hour_params[['Ticurr', 'Text']].apply(euler_dist, axis=1, arg=(Ti0, Te0))
    min_index = numpy.argmin(euler_param)

    #return hour_params.mean()
    #if usage == 4:
    #    print valid_params, hour
    #    sys.exit(1)

    return hour_params.loc[min_index]


# print current_int_temperature, current_ext_temperature, set_temperature
def run_algo(tidx, Tcurrint, Tscurr, Tset, Text, predicted_params):
    df_pred = pandas.DataFrame(Text).loc[tidx:]

    df_pred.loc[:, "T_pred"] = 0
    df_pred.loc[:, "S_pred"] = 0

    df_pred.loc[tidx, "T_pred"] = Tcurrint
    df_pred.loc[tidx, "S_pred"] = Tscurr

    alpha = predicted_params["alpha"]
    beta = predicted_params["beta"]
    epsilon = predicted_params["epsilon"]
    delta_on = predicted_params["delta_on"]
    delta_off = predicted_params["delta_off"]

    # T_off = predicted_params["Tl"]
    # T_on = predicted_params["Tu"]

    T_off = Tset + delta_off
    T_on = Tset + delta_on
    print alpha, beta, epsilon, Tset, delta_on, delta_off

    for idx in df_pred.index:
        p1 = df_pred.loc[idx, "ext_temperature"]# - df_pred.loc[idx, "T_pred"]
        p2 = df_pred.loc[idx, "S_pred"]

        df_pred.loc[idx+1, "T_pred"] = df_pred.loc[idx, "T_pred"] + (alpha * p1) + (beta * p2) + epsilon

        Tintc = df_pred.loc[idx, "T_pred"]
        Tintn = df_pred.loc[idx+1, "T_pred"]
        if Tintn <= Tintc:
            if Tintn > T_off:
                df_pred.loc[idx+1, "S_pred"] = 1
            elif Tintn <= T_off:
                df_pred.loc[idx+1, "S_pred"] = 0

        elif Tintn > Tintc:
            if Tintn <= T_on:
                df_pred.loc[idx+1, "S_pred"] = 0
            elif Tintn > T_on:
                df_pred.loc[idx+1, "S_pred"] = 1

    return df_pred

# run_algo_classifier_model_1 is the first thermal model implemented in PACMAN which is 
# T_int[t+1] - T_int[t] = alpha * (T_ext[t] - T_int[t]) + beta * S[t] + gamma * (T_ext[t]/30) + epsilon
# T_ext is divided by 30 to normalise it with other features
# Terminology:
#   alpha = leakage rate
#   beta = cooling rate
#   gamma, epsilon = coefficient of thermal noise
# Input:
#   df_pred - Prediction frame
#   Tset - Set temperature for the duration
#   predicted_params - learned parameters
#   classifier - learned classifier to predict AC compressor state
# Output:
#   df_pred - predicted temperature and AC compressor state for the usage
def run_algo_classifier_model_1(df_pred, Tset, predicted_params, classifier):

    # Start and stop time of AC 
    tidx = df_pred[df_pred["status"] == 1].index[0]
    lidx = df_pred[df_pred["status"] == 1].tail(1).index[0]

    # Extract AC usage data
    df_pred = df_pred[tidx:lidx]
    Ticurr = df_pred["int_temperature"].values[0]       # Current Room temperature
    Tscurr = df_pred["status"].values[0]                # Current AC state

    # Initialise an empty frame
    df_pred.loc[:, "T_pred"] = 0        # Predicted temperature
    df_pred.loc[:, "S_pred"] = 0        # Predicted AC compressor state

    df_pred.loc[tidx, "T_pred"] = df_pred.loc[tidx, "int_temperature"]
    df_pred.loc[tidx, "S_pred"] = df_pred.loc[tidx, "status"]

    # Learned parameters
    alpha = predicted_params["alpha"]
    beta = predicted_params["beta"]
    gamma = predicted_params["gamma"]
    epsilon = predicted_params["epsilon"]

    # Learned classifier
    clf = joblib.load('../Results/Prediction/models/' + classifier + '/clf.pkl')

    # Feature set
    v = Ticurr
    text = df_pred["ext_temperature"].values[0]
    p1 = text - v
    p2 = Tscurr
    p3 = numpy.divide(text, 30)

    # Predict for whole duration
    for idx in df_pred.index:

        # Predicted temperature difference
        diff = (alpha * p1) + (beta * p2) + (gamma * p3) + epsilon

        # Predicted temperature
        df_pred.loc[idx+1, "T_pred"] = v + diff

        # Update feature vector
        v = df_pred.loc[idx+1, "T_pred"]
        text = df_pred.loc[idx+1, "ext_temperature"]
        p1 = text - v

        p2 = clf.predict([v - Tset, diff])[0]
        p3 = numpy.divide(text, 30)

        # Predicted AC compressor state
        df_pred.loc[idx+1, "S_pred"] = p2

    # Return predicted frame
    return df_pred[:-1]


# run_algo_classifier_model_2 is the second thermal model implemented in PACMAN which is 
# T_int[t+1] - T_int[t] = alpha * (T_ext[t] - T_int[t]) + beta * S[t] + epsilon
# Terminology:
#   alpha = leakage rate
#   beta = cooling rate
#   epsilon = coefficient of thermal noise
# Input:
#   df_pred - Prediction frame
#   Tset - Set temperature for the duration
#   predicted_params - learned parameters
#   classifier - learned classifier to predict AC compressor state
# Output:
#   df_pred - predicted temperature and AC compressor state for the usage
def run_algo_classifier_model_2(df_pred, Tset, predicted_params, classifier):

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

    # Learned parameters
    alpha = predicted_params["alpha"]
    beta = predicted_params["beta"]
    epsilon = predicted_params["epsilon"]

    # Learned classifier
    clf = joblib.load('../Results/Prediction/models/' + classifier + '/clf.pkl')

    # Feature set
    v = Ticurr
    text = df_pred["ext_temperature"].values[0]
    p1 = text - v
    p2 = Tscurr

    # Predict for whole duration
    for idx in df_pred.index:

        # Predicted temperature difference
        diff = (alpha * p1) + (beta * p2) + epsilon

        # Predicted temperature
        df_pred.loc[idx+1, "T_pred"] = v + diff

        # Update feature vector
        v = df_pred.loc[idx+1, "T_pred"]
        text = df_pred.loc[idx+1, "ext_temperature"]

        p1 = numpy.divide(text, 30)
        p2 = clf.predict([v - Tset, diff])[0]

        # Predicted AC compressor state
        df_pred.loc[idx+1, "S_pred"] = p2

    # Return predicted frame
    return df_pred[:-1]



