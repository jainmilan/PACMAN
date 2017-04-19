import sys
import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.externals import joblib


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


# print current_int_temperature, current_ext_temperature, set_temperature
def run_algo_classifier_model_1(df_pred, Tset, predicted_params, classifier):
    tidx = df_pred[df_pred["status"] == 1].index[0]
    lidx = df_pred[df_pred["status"] == 1].tail(1).index[0]

    df_pred = df_pred[tidx:lidx]
    Ticurr = df_pred["int_temperature"].values[0]
    Tscurr = df_pred["status"].values[0]

    df_pred.loc[:, "T_pred"] = 0
    df_pred.loc[:, "S_pred"] = 0

    df_pred.loc[tidx, "T_pred"] = df_pred.loc[tidx, "int_temperature"]
    df_pred.loc[tidx, "S_pred"] = df_pred.loc[tidx, "status"]

    alpha = predicted_params["alpha"]
    beta = predicted_params["beta"]
    gamma = predicted_params["gamma"]
    epsilon = predicted_params["epsilon"]

    clf = joblib.load('../Results/Prediction/models/' + classifier + '/clf.pkl')

    v = Ticurr
    text = df_pred["ext_temperature"].values[0]
    p1 = text - v
    p2 = Tscurr
    p3 = numpy.divide(text, 30)

    for idx in df_pred.index:
        # Predicted Impact
        diff = (alpha * p1) + (beta * p2) + (gamma * p3) + epsilon

        # Predicted Temperature
        df_pred.loc[idx+1, "T_pred"] = v + diff

        # Update Model Parameters
        v = df_pred.loc[idx+1, "T_pred"]
        text = df_pred.loc[idx+1, "ext_temperature"]
        p1 = text - v

        p2 = clf.predict([v - Tset, diff])[0]
        p3 = numpy.divide(text, 30)

        # Predicted Status
        df_pred.loc[idx+1, "S_pred"] = p2

    return df_pred[:-1]


# print current_int_temperature, current_ext_temperature, set_temperature
def run_algo_classifier_model_2(df_pred, Tset, predicted_params, classifier):
    tidx = df_pred[df_pred["status"] == 1].index[0]
    lidx = df_pred[df_pred["status"] == 1].tail(1).index[0]

    df_pred = df_pred[tidx:lidx]
    Ticurr = df_pred["int_temperature"].values[0]
    Tscurr = df_pred["status"].values[0]

    df_pred.loc[:, "T_pred"] = 0
    df_pred.loc[:, "S_pred"] = 0

    df_pred.loc[tidx, "T_pred"] = df_pred.loc[tidx, "int_temperature"]
    df_pred.loc[tidx, "S_pred"] = df_pred.loc[tidx, "status"]

    alpha = predicted_params["alpha"]
    beta = predicted_params["beta"]
    epsilon = predicted_params["epsilon"]

    clf = joblib.load('../Results/Prediction/models/' + classifier + '/clf.pkl')

    v = Ticurr
    text = df_pred["ext_temperature"].values[0]
    p1 = text - v
    p2 = Tscurr

    for idx in df_pred.index:
        # Predicted Impact
        diff = (alpha * p1) + (beta * p2) + epsilon

        # Predicted Temperature
        df_pred.loc[idx+1, "T_pred"] = v + diff

        # Update Model Parameters
        v = df_pred.loc[idx+1, "T_pred"]
        text = df_pred.loc[idx+1, "ext_temperature"]

        p1 = numpy.divide(text, 30)
        p2 = clf.predict([v - Tset, diff])[0]

        # Predicted Status
        df_pred.loc[idx+1, "S_pred"] = p2

    return df_pred[:-1]


# print current_int_temperature, current_ext_temperature, set_temperature
def run_regressor_classifier_model_1(df_pred, Tset, fregr, fclass):
    tidx = df_pred[df_pred["status"] == 1].index[0]
    lidx = df_pred[df_pred["status"] == 1].tail(1).index[0]

    df_pred = df_pred[tidx:lidx]
    Ticurr = df_pred["int_temperature"].values[0]
    Tscurr = df_pred["status"].values[0]

    df_pred.loc[:, "T_pred"] = 0
    df_pred.loc[:, "S_pred"] = 0

    df_pred.loc[tidx, "T_pred"] = df_pred.loc[tidx, "int_temperature"]
    df_pred.loc[tidx, "S_pred"] = df_pred.loc[tidx, "status"]

    # File not found error
    try:
        regr = joblib.load(fregr)
        clf = joblib.load(fclass)
    except Exception as error:
        print error

    v = Ticurr
    text = df_pred["ext_temperature"].values[0]
    p1 = text - v
    p2 = Tscurr
    p3 = numpy.divide(text, 30)

    for idx in df_pred.index:
        # Predicted Impact
        diff = regr.predict([p1, p2, p3])

        # Predicted Temperature
        df_pred.loc[idx+1, "T_pred"] = v + diff[0]

        # Update Model Parameters
        v = df_pred.loc[idx+1, "T_pred"]
        text = df_pred.loc[idx+1, "ext_temperature"]

        p1 = text - v
        p2 = clf.predict([v - Tset, diff])[0]
        p3 = numpy.divide(text, 30)

        # Predicted Status
        df_pred.loc[idx+1, "S_pred"] = p2

    return df_pred[:-1]


# print current_int_temperature, current_ext_temperature, set_temperature
def run_regressor_classifier_model_2(df_pred, Tset, fregr, fclass):
    tidx = df_pred[df_pred["status"] == 1].index[0]
    lidx = df_pred[df_pred["status"] == 1].tail(1).index[0]

    df_pred = df_pred[tidx:lidx]
    Ticurr = df_pred["int_temperature"].values[0]
    Tscurr = df_pred["status"].values[0]

    df_pred.loc[:, "T_pred"] = 0
    df_pred.loc[:, "S_pred"] = 0

    df_pred.loc[tidx, "T_pred"] = df_pred.loc[tidx, "int_temperature"]
    df_pred.loc[tidx, "S_pred"] = df_pred.loc[tidx, "status"]

    # File not found error
    try:
        regr = joblib.load(fregr)
        clf = joblib.load(fclass)
    except Exception as error:
        print error

    v = Ticurr
    text = df_pred["ext_temperature"].values[0]
    p1 = text - v
    p2 = Tscurr

    for idx in df_pred.index:
        # Predicted Impact
        diff = regr.predict([p1, p2])

        # Predicted Temperature
        df_pred.loc[idx+1, "T_pred"] = v + diff[0]

        # Update Model Parameters
        v = df_pred.loc[idx+1, "T_pred"]
        text = df_pred.loc[idx+1, "ext_temperature"]

        p1 = text - v
        p2 = clf.predict([v - Tset, diff])[0]

        # Predicted Status
        df_pred.loc[idx+1, "S_pred"] = p2

    return df_pred[:-1]


# print current_int_temperature, current_ext_temperature, set_temperature
def run_regressor_classifier_model_3(df_pred, Tset, fregr, fclass):
    tidx = df_pred[df_pred["status"] == 1].index[0]
    lidx = df_pred[df_pred["status"] == 1].tail(1).index[0]

    df_pred = df_pred[tidx:lidx]
    Ticurr = df_pred["int_temperature"].values[0]
    Tscurr = df_pred["status"].values[0]

    df_pred.loc[:, "T_pred"] = 0
    df_pred.loc[:, "S_pred"] = 0

    df_pred.loc[tidx, "T_pred"] = df_pred.loc[tidx, "int_temperature"]
    df_pred.loc[tidx, "S_pred"] = df_pred.loc[tidx, "status"]

    # File not found error
    try:
        regr = joblib.load(fregr)
        clf = joblib.load(fclass)
    except Exception as error:
        print error

    v = Ticurr
    text = df_pred["ext_temperature"].values[0]
    p1 = v/25
    p2 = Tscurr
    p3 = text/25

    for idx in df_pred.index:
        # Predicted Impact
        diff = regr.predict([p1, p2, p3])

        # Predicted Temperature
        df_pred.loc[idx+1, "T_pred"] = v + diff[0]

        # Update Model Parameters
        v = df_pred.loc[idx+1, "T_pred"]
        text = df_pred.loc[idx+1, "ext_temperature"]

        p1 = v/25
        print v, Tset, diff
        p2 = clf.predict([v - Tset, diff[0]])[0]
        p3 = text/25

        # Predicted Status
        df_pred.loc[idx+1, "S_pred"] = p2

    return df_pred[:-1]


# print current_int_temperature, current_ext_temperature, set_temperature
def run_regressor_classifier_model_4(df_pred, Tset, fregr, fclass):
    tidx = df_pred[df_pred["status"] == 1].index[0]
    lidx = df_pred[df_pred["status"] == 1].tail(1).index[0]

    df_pred = df_pred[tidx:lidx]
    Ticurr = df_pred["int_temperature"].values[0]
    Tscurr = df_pred["status"].values[0]

    df_pred.loc[:, "T_pred"] = 0
    df_pred.loc[:, "S_pred"] = 0

    df_pred.loc[tidx, "T_pred"] = df_pred.loc[tidx, "int_temperature"]
    df_pred.loc[tidx, "S_pred"] = df_pred.loc[tidx, "status"]

    # File not found error
    try:
        regr = joblib.load(fregr)
        clf = joblib.load(fclass)
    except Exception as error:
        print error

    v = Ticurr
    text = df_pred["ext_temperature"].values[0]
    p1 = v
    p2 = Tscurr

    for idx in df_pred.index:
        # Predicted Impact
        v = regr.predict([p1, p2])[0]

        # Predicted Temperature
        df_pred.loc[idx+1, "T_pred"] = v

        # Update Model Parameters
        diff = v - df_pred.loc[idx, "T_pred"]

        p1 = v
        print Tset, v, diff
        p2 = clf.predict([v - Tset, diff])[0]

        # Predicted Status
        df_pred.loc[idx+1, "S_pred"] = p2

    return df_pred[:-1]


# print current_int_temperature, current_ext_temperature, set_temperature
def run_regressor_classifier_model_5(df_pred, Tset, fregr):
    tidx = df_pred[df_pred["status"] == 1].index[0]
    lidx = df_pred[df_pred["status"] == 1].tail(1).index[0]

    df_pred = df_pred[tidx:lidx]
    Ticurr = df_pred["int_temperature"].values[0]
    Tscurr = df_pred["status"].values[0]

    df_pred.loc[:, "T_pred"] = 0
    df_pred.loc[:, "S_pred"] = 0

    df_pred.loc[tidx, "T_pred"] = df_pred.loc[tidx, "int_temperature"]
    df_pred.loc[tidx, "S_pred"] = df_pred.loc[tidx, "status"]

    # File not found error
    try:
        regr = joblib.load(fregr)
    except Exception as error:
        print error

    v = Ticurr
    p1 = v
    p2 = Tscurr

    T_on = Tset + 2
    T_off = Tset

    for idx in df_pred.index:
        # Predicted Impact
        v = regr.predict([p1, p2])[0]

        # Predicted Temperature
        df_pred.loc[idx+1, "T_pred"] = v

        # Update Model Parameters
        diff = v - df_pred.loc[idx, "T_pred"]

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

        # Predicted Status
        df_pred.loc[idx+1, "S_pred"] = p2

    return df_pred[:-1]
