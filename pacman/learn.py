import os
import sys
import time
import glob
import numpy
import pandas
from .models import lr, svm, lp, knn, decision_tree, random_forest
from model import run_classifier, run_regressor


#<<-------------------------------------------------- Data Generator ----------------------------------------------->>#
def get_time(date_str, time_str):
    # Get time object from the string
    time_object = pandas.to_datetime(date_str + " " + time_str, dayfirst=True).tz_localize('Asia/Kolkata')

    # Convert to Numpy Object
    val_dt = numpy.datetime64(time_object)

    return val_dt


def update_param(df_act, df_learned, count):
    if df_act.empty:
        return df_learned

    df_act["alpha"] = ((df_act["alpha"])*(count-1) + df_learned["alpha"])/count
    df_act["beta"] = ((df_act["beta"])*(count-1) + df_learned["beta"])/count
    df_act["gamma"] = ((df_act["gamma"])*(count-1) + df_learned["gamma"])/count
    df_act["epsilon"] = ((df_act["epsilon"])*(count-1) + df_learned["epsilon"])/count

    return df_act


def learning_df(usages, last_usage, data, sampling_rate, pusages=-1):
    df = pandas.DataFrame()
    if last_usage == -1:
        return df

    elif last_usage == 0:
        sdate = usages["sdate"].iloc[last_usage]
        stime = usages["stime"].iloc[last_usage]

        Tset = usages["Tset"].iloc[last_usage]

        sidx = get_time(sdate, stime)
        df_temp = data.loc[:sidx]
        df_temp.loc[:, "Tset"] = Tset
        df_temp = df_temp.resample(sampling_rate, how='first')
        if len(df_temp) < 30:
            return df
        else:
            df = df_temp.copy()

    else:
        if (last_usage < pusages) or (pusages == -1):
            fusage = 0
        else:
            fusage = last_usage - pusages

        for usage in range(fusage, last_usage):
            sdate = usages["sdate"].iloc[usage]
            stime = usages["stime"].iloc[usage]
            sidx = get_time(sdate, stime)

            fdate = usages["fdate"].iloc[usage]
            ftime = usages["ftime"].iloc[usage]
            fidx = get_time(fdate, ftime)

            Tset = usages["Tset"].iloc[usage]
            check = usages["type"].iloc[usage]

            if check == -1:
                continue

            df_temp = data.loc[sidx:fidx]
            #print df_temp, sidx, fidx
            df_temp.loc[:, "Tset"] = Tset

            df_temp = df_temp.resample(sampling_rate, how='first')
            df = pandas.concat([df, df_temp])

    if len(df) < 30:
        return pandas.DataFrame()

    df = df.dropna()

    return df


#<<-------------------------------------------------- Classifiers ----------------------------------------------->>#
def learn_classifier_model_1(df, classifier, class_dir):
    data_size = len(df)

    df["idiff"] = df.int_temperature.diff()
    df["isdiff"] = df.int_temperature - df.Tset

    # Generate feature matrix for the model
    feature_matrix_classify = pandas.DataFrame()
    feature_matrix_classify["isdiff"] = df.tail(data_size-1).reset_index()["isdiff"]
    feature_matrix_classify["idiff"] = df.tail(data_size-1).reset_index()["idiff"]

    # Generate output matrix for the model
    Y_classify = df.tail(data_size-1).reset_index().status

    # Run classifier over the model
    done = run_classifier(feature_matrix_classify, Y_classify, classifier, 1, class_dir)

    return done


#<<-------------------------------------------------- Regressors ----------------------------------------------->>#
def learn_regressor_model_1(df, regressor, regress_dir):
    data_size = len(df)

    # Internal and External Temperature
    Tint = df['int_temperature'].values
    Text = df['ext_temperature'].values
    Text_norm = numpy.divide(Text, 30)
    status = df['status'].values

    # Parameters for our model
    diff_int_temp = numpy.ediff1d(Tint)

    diff_temperature = Text - Tint

    # Matrix for learning the Model
    m = data_size - 1
    feature_matrix_regress = numpy.zeros((m, 3))
    feature_matrix_regress[:, 0] = diff_temperature[0:-1]
    feature_matrix_regress[:, 1] = status[0:-1]
    feature_matrix_regress[:, 2] = Text_norm[0:-1]

    Y_regress = diff_int_temp

    # Run classifier over the model
    done = run_regressor(feature_matrix_regress, Y_regress, regressor, 1, regress_dir)

    return done


def learn_regressor_model_2(df, regressor, regress_dir):
    data_size = len(df)

    # Internal and External Temperature
    Tint = df['int_temperature'].values
    Text = df['ext_temperature'].values
    status = df['status'].values

    # Parameters for our model
    diff_int_temp = numpy.ediff1d(Tint)
    diff_temperature = Text - Tint

    # Matrix for learning the Model
    m = data_size - 1
    feature_matrix_regress = numpy.zeros((m, 2))
    feature_matrix_regress[:, 0] = diff_temperature[0:-1]
    feature_matrix_regress[:, 1] = status[0:-1]

    # Output Matrix
    Y_regress = diff_int_temp

    # Run classifier over the model
    done = run_regressor(feature_matrix_regress, Y_regress, regressor, 2, regress_dir)

    return done


def learn_regressor_model_3(df, regressor, regress_dir):
    data_size = len(df)

    # Internal and External Temperature
    Tint = df['int_temperature'].values
    Tint_norm = numpy.divide(Tint, 25)
    Text = df['ext_temperature'].values
    Text_norm = numpy.divide(Text, 25)
    status = df['status'].values

    # Parameters for our model
    diff_int_temp = numpy.ediff1d(Tint)

    # Matrix for learning the Model
    m = data_size - 1
    feature_matrix_regress = numpy.zeros((m, 3))
    feature_matrix_regress[:, 0] = Tint_norm[0:-1]
    feature_matrix_regress[:, 1] = status[0:-1]
    feature_matrix_regress[:, 2] = Text_norm[0:-1]

    Y_regress = diff_int_temp

    # Run classifier over the model
    done = run_regressor(feature_matrix_regress, Y_regress, regressor, 3, regress_dir)

    return done


def learn_regressor_model_4(df, regressor, regress_dir):
    data_size = len(df)

    # Internal and External Temperature
    Tint = df['int_temperature'].values
    status = df['status'].values

    # Matrix for learning the Model
    m = data_size - 1
    feature_matrix_regress = numpy.zeros((m, 2))
    feature_matrix_regress[:, 0] = Tint[0:-1]
    feature_matrix_regress[:, 1] = status[0:-1]

    Y_regress = Tint[1:]

    # Run classifier over the model
    done = run_regressor(feature_matrix_regress, Y_regress, regressor, 4, regress_dir)

    return done


def learn_regressor_model_5(df, regressor, regress_dir):
    data_size = len(df)

    # Internal and External Temperature
    Tint = df['int_temperature'].values
    status = df['status'].values

    # Matrix for learning the Model
    m = data_size - 1
    feature_matrix_regress = numpy.zeros((m, 2))
    feature_matrix_regress[:, 0] = Tint[0:-1]
    feature_matrix_regress[:, 1] = status[0:-1]

    Y_regress = Tint[1:]

    # Run classifier over the model
    done = run_regressor(feature_matrix_regress, Y_regress, regressor, 5, regress_dir)

    return done


#<<-------------------------------------------- Hourly Implementation ----------------------------------------------->>#
def learn_parameters_hourly(df, room, usage, Tset, param_file, con, coff, model):
    df_param = pandas.DataFrame()
    if os.path.isfile(param_file):
        df_param = pandas.read_csv(param_file, index_col=["ID"])
        if len(df_param[df_param["uid"] == str(room)+"_"+str(usage)]) > 0:
            return

    df_grouped = df.groupby(df.index.hour)

    if model == "lr":
        df_temp = df_grouped.apply(lr.run_4)
    elif model == "svm":
        df_temp = df_grouped.apply(svm.run_2)
    elif model == "lp":
        df_temp = df_grouped.apply(lp.run)

    df_temp.index.name = "hour"
    df_temp = df_temp.reset_index()

    df_temp.loc[:, "uid"] = str(room) + "_" + str(usage)
    df_temp.loc[:, "Tset"] = Tset

    df_temp.loc[:, "delta_on"] = con - Tset
    df_temp.loc[:, "delta_off"] = coff - Tset

    df_param = df_param.append(df_temp, ignore_index=True)
    df_param.to_csv(param_file, index_label="ID")


#<<-------------------------------------------------- Deprecated ----------------------------------------------->>#
def learn_parameters(data_file, r):
    training_data = pandas.read_csv(data_file, index_col=0)

    # Preprocessing
    temp = len(training_data)
    training_data = training_data[r:]
    training_data = training_data[:temp - r]

    data_size = len(training_data.index.values)

    # Timestamps at which AC compressor turns On. In second line, we are extracting invalid entries such as 0
    on_time_raw = numpy.unique(training_data['onTime'].values)
    on_time = numpy.delete(on_time_raw, numpy.where(on_time_raw == 0)[0])

    # Timestamps at which AC compressor turns Off. In second line, we are extracting invalid entries such as 0
    off_time_raw = numpy.unique(training_data['offTime'].values)
    off_time = numpy.delete(off_time_raw, numpy.where(off_time_raw == 0)[0])

    # Internal and External Temperature
    int_temperature = training_data['int_temperature'].values
    ext_temperature = training_data['ext_temperature'].values

    # Parameters for our model
    diff_int_temp = numpy.ediff1d(int_temperature)

    diff_temperature = ext_temperature - int_temperature
    status = training_data['status'].values

    # Matrix for learning the Model
    m = data_size - 1
    feature_matrix = numpy.zeros((m, 2))
    feature_matrix[:, 0] = diff_temperature[0:-1]
    feature_matrix[:, 1] = status[0:-1]

    Y_linear = diff_int_temp

    #alpha, beta, epsilon = lr.run(feature_matrix, Y_linear)
    alpha, beta, epsilon = svm.run(feature_matrix, Y_linear)

    Tcurrint = int_temperature[0]
    Tcurrext = ext_temperature[0]
    Tset = numpy.unique(training_data["setTemperature"])[0]

    Tl = numpy.mean(int_temperature) - numpy.std(int_temperature)
    Tu = numpy.mean(int_temperature) + numpy.std(int_temperature)

    phase = 0
    if Tcurrint < numpy.mean(ext_temperature):
        phase = 1

    parameters = {'alpha': alpha, 'beta': beta, 'epsilon': epsilon, 'Tcurrint': Tcurrint, \
                  'Tcurrext': Tcurrext, 'Tset': Tset, 'Tl': Tl, 'Tu': Tu, 'Phase': phase}
    return parameters
