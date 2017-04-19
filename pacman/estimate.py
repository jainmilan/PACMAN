import sys
import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# get_status_clustered function estimates the AC compressor state by only using the room 
# temperature information. PACMAN implements K-means clustering for the analysis
# Input:
#   df - input AC usage data
#   Tset - AC set temperature
# Output:
#   df - output frame with the estimation of AC compressor state
def get_status_clustered(df, Tset):

    # Room temperature
    Tint = df["int_temperature"]

    # K-Mean clustering where k=3
    est = KMeans(n_clusters=3)

    # Feature space
    X = numpy.ediff1d(Tint.values)
    X = X.reshape((len(X)),1)

    # Learn the clusters
    est.fit(X)

    # Fetch label of each cluster
    max_min_labels = [numpy.argmin(est.cluster_centers_), numpy.argmax(est.cluster_centers_)]
    labels = est.labels_

    # Labels future event
    df.loc[:-1, "labels"] = labels 

    # Previous Event is On
    val = 1 

    # Compressor Off Event
    search = max_min_labels[1] 

    # Initialization
    df.loc[:, "status_est"] = 0
    df.loc[:, "status_rbias"] = 0

    # Ignore initial two instances
    for idx in df[:2].index:
        df.loc[idx, "status_est"] = df.loc[idx, "status"]
        df.loc[idx, "status_rbias"] = df.loc[idx, "status"]

    # Estimate AC compressor state
    for idx in df[2:-2].index:
        df.loc[idx, "status_rbias"] = 1
        if df.loc[idx, "labels"] == search:     # If found what searching for
            if (search == max_min_labels[0]):   # If off then search for on
                val = 1
                search = max_min_labels[1]
            elif (search == max_min_labels[1]): # If on then search for off
                val = 0
                search = max_min_labels[0]
        df.loc[idx, "status_est"] = val

    # Ignore last two instances
    for idx in df[-2:].index:
        df.loc[idx, "status_est"] = df.loc[idx, "status"]
        df.loc[idx, "status_rbias"] = df.loc[idx, "status"]

    # Return the data frame
    return df