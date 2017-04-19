__author__ = 'milan'

# Inbuilt libraries
import os
import sys
import time
import glob
import numpy
import pandas

# Custom modules
from model import run_classifier, run_regressor


#<<-------------------------------------------------- Data Generator ----------------------------------------------->>#
def get_time(date_str, time_str):
    # Get time object from the string
    time_object = pandas.to_datetime(date_str + " " + time_str, dayfirst=True).tz_localize('Asia/Kolkata')

    # Convert to Numpy Object
    val_dt = numpy.datetime64(time_object)

    return val_dt


# learning_df function combines all the historical usages and generates a common frame
# to learn a regressor and a classifier
# Input:
#   usages - total number of valid usages
#   last_usage - last predicted usage
#   data - complete temperature and energy consumption data of the room
#   sampling_rate - sampling rate of data as input by the user
#   pusages - size of training dataset (by default - all)
# Output:
#   df - dataframe comprising of all the historical data
def learning_df(usages, last_usage, data, sampling_rate, pusages=-1):

    # Initialize an empty frame to be returned
    df = pandas.DataFrame()

    # Invalid AC usage
    if last_usage == -1:
        return df

    # First AC usage
    elif last_usage == 0:
        # Start and stop time of last predicted AC usage
        sdate = usages["sdate"].iloc[last_usage]
        stime = usages["stime"].iloc[last_usage]

        # Set temperature of last predicted AC usage
        Tset = usages["Tset"].iloc[last_usage]

        # Convert string to datetime
        sidx = get_time(sdate, stime)

        # Subset the dataset till the time AC starts
        df_temp = data.loc[:sidx]
        df_temp.loc[:, "Tset"] = Tset

        # Resample the selected dataset
        df_temp = df_temp.resample(sampling_rate, how='first')

        # Check if data is atleast of one-hour duration
        if len(df_temp) < 30:
            return df
        else:
            df = df_temp.copy()

    # Any other AC usage
    else:

        # The id of last predicted usage should atleast be equal to pusages else select 
        # complete data
        if (last_usage < pusages) or (pusages == -1):
            fusage = 0
        else:
            fusage = last_usage - pusages

        # Iterate through all usages between first AC usage and last predicted AC usage
        for usage in range(fusage, last_usage):

            # Start time of the selected usage
            sdate = usages["sdate"].iloc[usage]
            stime = usages["stime"].iloc[usage]
            sidx = get_time(sdate, stime)

            # Stop time of the selected usage
            fdate = usages["fdate"].iloc[usage]
            ftime = usages["ftime"].iloc[usage]
            fidx = get_time(fdate, ftime)

            # Set temperature and usage validity check
            Tset = usages["Tset"].iloc[usage]

            check = usages["type"].iloc[usage]
            if check == -1:
                continue

            # Subset the dataset
            df_temp = data.loc[sidx:fidx]
            df_temp.loc[:, "Tset"] = Tset

            # Resample to desired frequency and merge with the combined data frame
            df_temp = df_temp.resample(sampling_rate, how='first')
            df = pandas.concat([df, df_temp])

    # Return empty frame if data size is less than an hour
    if len(df) < 30:
        return pandas.DataFrame()

    # Remove NA values
    df = df.dropna()

    # Return the dataframe
    return df


#<<-------------------------------------------------- Classifiers ----------------------------------------------->>#

# learn_classifier_model_1 is the classifier of PACMAN to predict AC compressor state based on:
#   T_int[t+1] - T_int[t]
#   T_int[t+1] - T_set
# Input:
#   df - Data frame
#   classifier - machine learning algorithm to use as classifier
#   class_dir - directory to save the learned parameters of the classifier
# Output:
#   done - status of learning the model
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

# learn_regressor_model_1 is the first thermal model implemented in PACMAN which is 
# T_int[t+1] - T_int[t] = alpha * (T_ext[t] - T_int[t]) + beta * S[t] + gamma * (T_ext[t]/30) + epsilon
# T_ext is divided by 30 to normalise it with other features
# Terminology:
#   alpha = leakage rate
#   beta = cooling rate
#   gamma, epsilon = coefficient of thermal noise
# Input:
#   df - Data frame
#   regressor - machine learning algorithm to use as regressor
#   regress_dir - directory to save the learned parameters of the regressor
# Output:
#   done - status of learning the model
def learn_regressor_model_1(df, regressor, regress_dir):

    # Length of training data frame
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

    # Run regressor over the model
    done = run_regressor(feature_matrix_regress, Y_regress, regressor, 1, regress_dir)

    # Return the status
    return done

# learn_regressor_model_2 is the second thermal model implemented in PACMAN which is 
# T_int[t+1] - T_int[t] = alpha * (T_ext[t] - T_int[t]) + beta * S[t] + epsilon
# Terminology:
#   alpha = leakage rate
#   beta = cooling rate
#   epsilon = coefficient of thermal noise
# Input:
#   df - Data frame
#   regressor - machine learning algorithm to use as regressor
#   regress_dir - directory to save the learned parameters of the regressor
# Output:
#   done - status of learning the model
def learn_regressor_model_2(df, regressor, regress_dir):

    # Length of training data frame
    data_size = len(df)

    # Internal and external temperature
    Tint = df['int_temperature'].values
    Text = df['ext_temperature'].values
    status = df['status'].values

    # Parameters for our model
    diff_int_temp = numpy.ediff1d(Tint)
    diff_temperature = Text - Tint

    # Matrix for learning the model
    m = data_size - 1
    feature_matrix_regress = numpy.zeros((m, 2))
    feature_matrix_regress[:, 0] = diff_temperature[0:-1]
    feature_matrix_regress[:, 1] = status[0:-1]

    # Output matrix
    Y_regress = diff_int_temp

    # Run classifier over the model
    done = run_regressor(feature_matrix_regress, Y_regress, regressor, 2, regress_dir)

    # Return the status
    return done


# learn_regressor_model_3 is the third thermal model implemented in PACMAN which is 
# T_int[t+1] - T_int[t] = alpha * (T_int[t]/25) + beta * S[t] + gamma * (T_ext[t]/25) + epsilon
# Terminology:
#   alpha = leakage rate
#   beta = cooling rate
#   gamma, epsilon = coefficient of thermal noise
# Input:
#   df - Data frame
#   regressor - machine learning algorithm to use as regressor
#   regress_dir - directory to save the learned parameters of the regressor
# Output:
#   done - status of learning the model
def learn_regressor_model_3(df, regressor, regress_dir):

    # Length of training data frame
    data_size = len(df)

    # Internal and external temperature
    Tint = df['int_temperature'].values
    Tint_norm = numpy.divide(Tint, 25)
    Text = df['ext_temperature'].values
    Text_norm = numpy.divide(Text, 25)
    status = df['status'].values

    # Parameters for our model
    diff_int_temp = numpy.ediff1d(Tint)

    # Matrix for learning the model
    m = data_size - 1
    feature_matrix_regress = numpy.zeros((m, 3))
    feature_matrix_regress[:, 0] = Tint_norm[0:-1]
    feature_matrix_regress[:, 1] = status[0:-1]
    feature_matrix_regress[:, 2] = Text_norm[0:-1]

    Y_regress = diff_int_temp

    # Run classifier over the model
    done = run_regressor(feature_matrix_regress, Y_regress, regressor, 3, regress_dir)

    # Return the status
    return done


# learn_regressor_model_4 is the fourth thermal model implemented in PACMAN which is 
# T_int[t+1] - T_int[t] = alpha * T_int[t] + beta * S[t] + epsilon
# Terminology:
#   alpha = leakage rate
#   beta = cooling rate
#   epsilon = coefficient of thermal noise
# Input:
#   df - Data frame
#   regressor - machine learning algorithm to use as regressor
#   regress_dir - directory to save the learned parameters of the regressor
# Output:
#   done - status of learning the model
def learn_regressor_model_4(df, regressor, regress_dir):

    # Length of training data frame
    data_size = len(df)

    # Internal and external temperature
    Tint = df['int_temperature'].values
    status = df['status'].values

    # Matrix for learning the model
    m = data_size - 1
    feature_matrix_regress = numpy.zeros((m, 2))
    feature_matrix_regress[:, 0] = Tint[0:-1]
    feature_matrix_regress[:, 1] = status[0:-1]

    Y_regress = Tint[1:]

    # Run classifier over the model
    done = run_regressor(feature_matrix_regress, Y_regress, regressor, 4, regress_dir)

    # Return the status
    return done


# learn_regressor_model_5 uses the same thermal model as the fourth thermal model but 
# uses threshold based state prediction instead of a learned classifier. Following is the implemented 
# thermal model for the scenario.
# T_int[t+1] - T_int[t] = alpha * T_int[t] + beta * S[t] + epsilon
# Terminology:
#   alpha = leakage rate
#   beta = cooling rate
#   epsilon = coefficient of thermal noise
# Input:
#   df - Data frame
#   regressor - machine learning algorithm to use as regressor
#   regress_dir - directory to save the learned parameters of the regressor
# Output:
#   done - status of learning the model
def learn_regressor_model_5(df, regressor, regress_dir):

    # Length of training data frame
    data_size = len(df)

    # Internal and external temperature
    Tint = df['int_temperature'].values
    status = df['status'].values

    # Matrix for learning the model
    m = data_size - 1
    feature_matrix_regress = numpy.zeros((m, 2))
    feature_matrix_regress[:, 0] = Tint[0:-1]
    feature_matrix_regress[:, 1] = status[0:-1]

    Y_regress = Tint[1:]

    # Run classifier over the model
    done = run_regressor(feature_matrix_regress, Y_regress, regressor, 5, regress_dir)

    # Return the status
    return done