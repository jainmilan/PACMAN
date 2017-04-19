import sys
import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def get_status(Tint, r):
    # Preprocessing
    temp = len(Tint)
    Tint = Tint[r:]
    Tint = Tint[:temp - r]

    duration = len(Tint)
    Tstatus = numpy.zeros(duration)

    c = 0
    THRESHOLD = 45
    for t in range(duration):
        #		print c, Tstatus[t]
        if t == 0:
            prev_status = Tstatus[t]
        else:
            if Tint[t] <= Tint[t - 1]:
                if prev_status == 1:
                    if c >= THRESHOLD:
                        Tstatus[t] = 1
                    else:
                        Tstatus[t] = Tstatus[t - 1]
                    c = c + 1
                else:
                    Tstatus[t] = Tstatus[t - 1]
                    c = 0
                prev_status = 1

            elif Tint[t] > Tint[t - 1]:
                if prev_status == 0:
                    if c >= THRESHOLD:
                        Tstatus[t] = 0
                    else:
                        Tstatus[t] = Tstatus[t - 1]
                    c = c + 1
                else:
                    Tstatus[t] = Tstatus[t - 1]
                    c = 0
                prev_status = 0

    return Tstatus


def get_status_clustered(df, Tset):
    Tint = df["int_temperature"]
    est = KMeans(n_clusters=3)

    X = numpy.ediff1d(Tint.values)
    X = X.reshape((len(X)),1)
    est.fit(X)

    max_min_labels = [numpy.argmin(est.cluster_centers_), numpy.argmax(est.cluster_centers_)]
    labels = est.labels_

    df.loc[:-1, "labels"] = labels # Labels future event

    val = 1 # Previous Event is On
    search = max_min_labels[1] # Compressor Off Event

    # Initialization
    df.loc[:, "status_est"] = 0
    df.loc[:, "status_rbias"] = 0

    for idx in df[:2].index:
        df.loc[idx, "status_est"] = df.loc[idx, "status"]
        df.loc[idx, "status_rbias"] = df.loc[idx, "status"]

    for idx in df[2:-2].index:
        df.loc[idx, "status_rbias"] = 1
        if df.loc[idx, "labels"] == search:
            if (search == max_min_labels[0]):
                val = 1
                search = max_min_labels[1]
            elif (search == max_min_labels[1]):
                val = 0
                search = max_min_labels[0]
        df.loc[idx, "status_est"] = val

    for idx in df[-2:].index:
        df.loc[idx, "status_est"] = df.loc[idx, "status"]
        df.loc[idx, "status_rbias"] = df.loc[idx, "status"]

    return df