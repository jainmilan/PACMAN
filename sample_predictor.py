__author__ = 'milan'

# Import standard libraries
import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import pacman library
from pacman import predict, learn, stats

# String to datetime conversion
def get_time(date_str, time_str):

    # Get time object from the string
    time_object = pd.to_datetime(date_str + " " + time_str, dayfirst=True).tz_localize('Asia/Kolkata')

    # Convert to Numpy Object
    val_dt = np.datetime64(time_object)

    return val_dt

# Main function
if __name__ == "__main__":

    # Data set to analyse
    data_dir = "../Data/Apartment/"
    rooms = ["01000000", "02020302", "04050101", "05050203", "06060206", "07070304", "08090104"]
    
    # To learn the parameters of a thermal model, PACMAN requires a regressor which are defined in 
    # models directory. To access any model, user need to call model.py. 
    regressor = "lr"
    regr_type = 1

    # PACMAN calls a classifier to estimate the AC compressor state based on the room temperature data.
    # We have implemented five classifiers in our current implementation which are available in 
    # model directory and can be accessed through model.py
    classifier = 'rf'
    class_type = 1

    # Desired sampling rate of data
    sampling_rate = '2T'
    
    # Number of historical usages to learn the regressor and classifier. List can attempt multiple such usages    
    pusages_bin = [10]

    # Frame to store the statistics generated during the analysis.
    df_stats = pd.DataFrame()

    count = 0
    
    # Iterate through the data from each room
    for room in rooms:

        # Iterate through different number of historical usages. In testing, we are using 10 AC usages.
        for pusages in pusages_bin:

            # Variables to track the state of loop
            bias = 0            # Evaluation metric
            done_regr = None    # If learning through regression is complete
            done_class = None   # If learning through classifier is complete

            # Suffix to store the results
            suffix =  room + '/' + str(regr_type) + "_" + str(class_type) + '/' + regressor + "_" + classifier + \
                      "/pusages_" + str(pusages) + "/sr_" + str(sampling_rate) + "/"

            # Update directories with suffix
            dir_pred = "../Results/Prediction/" + suffix
            dir_learn = "../Results/Learning/" + suffix
            dir_estimate = "../Results/Estimation/" + suffix

            # Input Data
            usages = pd.read_csv(data_dir + room + "_Usages.csv", index_col=["id"]) # Meta Data
            data = pd.read_csv(data_dir + room + "_updated.csv", index_col=[0])     # Temperature and energy consumption data

            # Resample intermediate data to 1 second frequency for alignment
            idx = pd.to_datetime(data.index, unit='s').tz_localize('UTC').tz_convert('Asia/Kolkata')
            data.index = idx
            data = data.resample('1S', fill_method='bfill')

            # Preprocessing [Select only valid usages]
            usages = usages[usages["type"] == 1]

            # Meta information about the AC usage
            prated = data["prated"].unique()[0]                 # Rated power consumption of AC
            manufacturer = data["manufacturer"].unique()[0]     # AC Manufacturer

            # Start date and time
            sdate = usages["sdate"].iloc[0]
            stime = usages["stime"].iloc[0]
            sidx = get_time(sdate, stime)   # Convert from string to datetime

            # Difference between AC start time and current time
            fusage_diff = (sidx - data.index.values[0]) / np.timedelta64(1, 's')

            # Smoothen external temperature data
            if fusage_diff < 7200:  # Skip first AC usage if it lies within 2 hours
                sys.exit(1)
            else:
                data["ext_temperature_unsmooth"] = data["ext_temperature"]
                data["ext_temperature"] = pd.rolling_mean(data.ext_temperature, 7200)
                data = data.dropna()

            # Iterate over all the AC usages that occured in the room
            usage_bin = range(len(usages))
            for usage in usage_bin:

                # Skip invalid usages i.e. Abnormal Usages
                check = usages["type"].iloc[usage]
                if check == -1:
                    print "Invalid usage-%d" %(usage)
                    continue

                # Start date and time
                sdate = usages["sdate"].iloc[usage]
                stime = usages["stime"].iloc[usage]
                sidx = get_time(sdate, stime)

                # Stop date and time
                fdate = usages["fdate"].iloc[usage]
                ftime = usages["ftime"].iloc[usage]
                fidx = get_time(fdate, ftime)

                # Set temperature and actual usage ID
                Tset = usages["Tset"].iloc[usage]
                usage_act = usages["usage"].iloc[usage]

                # Extract current AC usage
                df = data.loc[sidx:fidx][["int_temperature", "ext_temperature", "power", "status"]]

                # Resample to the given sampling rate
                df_resampled = df.resample(sampling_rate, how='first')

                # Skip prediction for first usage
                if not done_regr is None:
                    
                    # Parameters learned for regressor and classifier (NA for first time prediction)
                    file_regr = dir_learn + 'models/regr/' + regressor + '_' + str(regr_type) + '.pkl'
                    file_classifier = dir_learn + 'models/class/' + classifier + '_' + str(class_type) + '.pkl'

                    # Predict temperature and AC compressor state
                    if regr_type == 1:
                        df_pred = predict.run_regressor_classifier_model_1(df_resampled, Tset, file_regr, file_classifier)
                    elif regr_type == 2:
                        df_pred = predict.run_regressor_classifier_model_2(df_resampled, Tset, file_regr, file_classifier)
                    elif regr_type == 3:
                        df_pred = predict.run_regressor_classifier_model_3(df_resampled, Tset, file_regr, file_classifier)
                    elif regr_type == 4:
                        df_pred = predict.run_regressor_classifier_model_4(df_resampled, Tset, file_regr, file_classifier)
                    elif regr_type == 5:
                        df_pred = predict.run_regressor_classifier_model_5(df_resampled, Tset, file_regr)

                    # Estimate bias in AC energy consumption
                    bias = bias + (df_pred.status.sum() - df_pred.S_pred.sum())

                    # Create directory to save prediction results
                    dir_predictions = dir_pred + "CSV/"
                    if not os.path.exists(dir_predictions):
                        os.makedirs(dir_predictions)

                    df_pred.to_csv(dir_predictions + "usage_" + str(usage_act) + ".csv")

                    # Generate prediction statistics and update the stat frame - df_stats
                    df_tempstats = stats.analyze(df_pred.status, df_pred.S_pred, room, usage, regr_type, regressor,
                                                 classifier, count, pusages, usage_act, Tset, prated, manufacturer)
                    df_stats = pd.concat([df_stats, df_tempstats])

                    # Count number of valid predictions
                    count = count + 1

                # Combined previous usages
                df_learn = learn.learning_df(usages=usages, last_usage=usage, data=data, sampling_rate = sampling_rate,
                                             pusages=pusages)

                # Ignore empty training dataframe
                if df_learn.empty:
                    print "Nothing to learn from usage-%d" %(usage)
                    continue

                # Directory to learn regression and classifier
                dir_regr = dir_learn + 'models/regr/'
                dir_class = dir_learn + 'models/class/'

                # Learn the regressor and the classifier
                if regr_type == 1:
                    done_regr = learn.learn_regressor_model_1(df_learn, regressor, dir_regr)
                    done_class = learn.learn_classifier_model_1(df_learn, classifier, dir_class)
                elif regr_type == 2:
                    done_regr = learn.learn_regressor_model_2(df_learn, regressor, dir_regr)
                    done_class = learn.learn_classifier_model_1(df_learn, classifier, dir_class)
                elif regr_type == 3:
                    done_regr = learn.learn_regressor_model_3(df_learn, regressor, dir_regr)
                    done_class = learn.learn_classifier_model_1(df_learn, classifier, dir_class)
                elif regr_type == 4:
                    done_regr = learn.learn_regressor_model_4(df_learn, regressor, dir_regr)
                    done_class = learn.learn_classifier_model_1(df_learn, classifier, dir_class)
                elif regr_type == 5:
                    done_regr = learn.learn_regressor_model_5(df_learn, regressor, dir_regr)

            # Create directory if doesn't exist and save the intermediatory stats
            dir_stats = dir_pred + "stats/"
            if not os.path.exists(dir_stats):
                os.makedirs(dir_stats)

            df_stats.to_csv(dir_stats + "stats.csv")

    # Create directory if doesn't exist and save the final stats
    dir_res = "results/"
    if not os.path.exists(dir_res):
        os.makedirs(dir_res)
    
    df_stats.to_csv(dir_res + "stats_10.csv")