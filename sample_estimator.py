__author__ = 'milan'

# Import standard libraries
import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import pacman library
from pacman import predict, learn, stats, estimate

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
    data_dir = "sample_data/"
    rooms = ["rx"]

    # Desired sampling rate of data
    sampling_rate = '2T'

    # Stats to evaluate the estimation accuracy
    df_est_stats = pd.DataFrame()

    # Iterate through the data from each room
    for room in rooms:

        # Suffix to store the results
        suffix = room + "/sr_" + str(sampling_rate) + "/"

        # Update directories with suffix
        dir_estimate = "results/estimation/" + suffix

        # Input Data
        usages = pd.read_csv(data_dir + room + "_meta.csv", index_col=["id"])
        data = pd.read_csv(data_dir + room + "_data.csv", index_col=[0])

        # Resample intermediate data to 1 second frequency for alignment
        idx = pd.to_datetime(data.index, unit='s').tz_localize('UTC').tz_convert('Asia/Kolkata')
        data.index = idx
        data = data.resample('1S', fill_method='bfill')

        # Preprocessing [Select only valid usages]
        usages = usages[usages["type"] == 1]

        # Meta information about the AC usage
        prated = data["prated"].unique()[0]
        manufacturer = data["manufacturer"].unique()[0]

        # Start date and time
        sdate = usages["sdate"].iloc[0]
        stime = usages["stime"].iloc[0]
        sidx = get_time(sdate, stime)

        # Difference between AC start time and current time
        fusage_diff = (sidx - data.index.values[0]) / np.timedelta64(1, 's')

        # Smoothen external temperature data
        if fusage_diff < 7200:  # Skip first AC usage if it lies within 2 hours
            print "Hello"
            sys.exit(1)
        else:
            data["ext_temperature_unsmooth"] = data["ext_temperature"]
            data["ext_temperature"] = pd.rolling_mean(data.ext_temperature, 7200)
            data = data.dropna()

        # Iterate over all the AC usages that occured in the room
        usage_bin = range(len(usages))
        for usage in usage_bin:

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

            # Preprocessing
            start_idx = df_resampled[df_resampled["status"] == 1].head(1).index.values[0]
            start_idx = df_resampled.index.values[df_resampled.index.get_loc(start_idx) - 1]

            stop_idx = df_resampled[df_resampled["status"] == 1].tail(1).index.values[0]
            stop_idx = df_resampled.index.values[df_resampled.index.get_loc(stop_idx) + 1]

            df_subset = df_resampled[start_idx:stop_idx]

            # Estimation
            if len(df_subset) < 5:
                continue

            # Generate cluster and estimate AC compressor state
            df_est = estimate.get_status_clustered(df_subset, Tset)

            # Generate estimation statistics and update the stat frame - df_est_stats, df_est_stats_rbias
            df_temp_est_stats = stats.analyze_est(df_est.status, df_est["status_est"], room, usage, usage_act,
                                                  Tset, prated, manufacturer)

            df_est_stats = pd.concat([df_est_stats, df_temp_est_stats])


        # Create directory if doesn't exist and save the intermediatory stats
        dir_stats = dir_estimate + "stats/"
        if not os.path.exists(dir_stats):
            os.makedirs(dir_stats)

        df_est_stats.to_csv(dir_stats + "stats_est.csv")

    # Create directory if doesn't exist and save the final stats
    dir_res = "results/"
    if not os.path.exists(dir_res):
        os.makedirs(dir_res)
    
    df_est_stats.to_csv(dir_res + "stats_est.csv")