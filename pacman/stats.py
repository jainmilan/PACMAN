__author__ = 'milan'

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

def analyze(actual, predicted, room, usage, model_type, regressor, classifier, count, pusages, usage_act, Tset,
            prated, manufacturer):
    stats = pd.DataFrame()

    E_act = actual.sum()
    E_est = predicted.sum()

    error = E_act - E_est
    error_abs = np.abs(error)
    acc = 100 - (error_abs*100.0)/(E_act)
    print acc
    prfs = precision_recall_fscore_support(actual.values, predicted.values)
    cm = confusion_matrix(actual.values, predicted.values)

    approach = count
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Room"] = room
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Usage"] = usage
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "UsageAct"] = usage_act
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "PUsages"] = pusages
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Approach"] = model_type
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Regressor"] = regressor
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Classifier"] = classifier
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "E_act"] = E_act
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "E_est"] = E_est
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Error"] = error
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Absolute Error"] = error_abs
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Accuracy"] = acc
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Tset"] = Tset
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "PRated"] = prated
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Manufacturer"] = manufacturer

    if len(prfs[0]) == 2:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Precision[0]"] = np.round(prfs[0][0]*100.0, 2)
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Precision[1]"] = np.round(prfs[0][1]*100.0, 2)
    else:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Precision[0]"] = -1
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Precision[1]"] = np.round(prfs[0][0]*100.0, 2)

    if len(prfs[1]) == 2:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Recall[0]"] = np.round(prfs[1][0]*100.0, 2)
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Recall[1]"] = np.round(prfs[1][1]*100.0, 2)
    else:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Recall[0]"] = -1
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Recall[1]"] = np.round(prfs[1][0]*100.0, 2)


    if len(prfs[2]) == 2:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "FScore[0]"] = np.round(prfs[2][0]*100.0, 2)
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "FScore[1]"] = np.round(prfs[2][1]*100.0, 2)
    else:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "FScore[0]"] = -1
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "FScore[1]"] = np.round(prfs[2][0]*100.0, 2)

    if len(cm) == 2:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[00]"] = cm[0][0]
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[01]"] = cm[0][1]
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[10]"] = cm[1][0]
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[11]"] = cm[1][1]
    else:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[00]"] = -1
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[01]"] = -1
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[10]"] = -1
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[11]"] = cm[0][0]

    return stats


def analyze_est(actual, predicted, room, usage, usage_act, Tset, prated, manufacturer):
    stats = pd.DataFrame()

    E_act = actual.sum()
    E_est = predicted.sum()

    error = E_act - E_est
    error_abs = np.abs(error)
    acc = 100 - (error_abs*100.0)/(E_act)
    print acc
    prfs = precision_recall_fscore_support(actual.values, predicted.values)
    cm = confusion_matrix(actual.values, predicted.values)

    approach = 1
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Room"] = room
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Usage"] = usage
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "UsageAct"] = usage_act
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "E_act"] = E_act
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "E_est"] = E_est
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Error"] = error
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Absolute Error"] = error_abs
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Accuracy"] = acc
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Tset"] = Tset
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "PRated"] = prated
    stats.loc[room + "_" + str(usage) + "_" + str(approach), "Manufacturer"] = manufacturer

    if len(prfs[0]) == 2:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Precision[0]"] = np.round(prfs[0][0]*100.0, 2)
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Precision[1]"] = np.round(prfs[0][1]*100.0, 2)
    else:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Precision[0]"] = -1
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Precision[1]"] = np.round(prfs[0][0]*100.0, 2)

    if len(prfs[1]) == 2:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Recall[0]"] = np.round(prfs[1][0]*100.0, 2)
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Recall[1]"] = np.round(prfs[1][1]*100.0, 2)
    else:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Recall[0]"] = -1
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "Recall[1]"] = np.round(prfs[1][0]*100.0, 2)


    if len(prfs[2]) == 2:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "FScore[0]"] = np.round(prfs[2][0]*100.0, 2)
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "FScore[1]"] = np.round(prfs[2][1]*100.0, 2)
    else:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "FScore[0]"] = -1
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "FScore[1]"] = np.round(prfs[2][0]*100.0, 2)

    if len(cm) == 2:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[00]"] = cm[0][0]
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[01]"] = cm[0][1]
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[10]"] = cm[1][0]
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[11]"] = cm[1][1]
    else:
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[00]"] = -1
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[01]"] = -1
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[10]"] = -1
        stats.loc[room + "_" + str(usage) + "_" + str(approach), "ConfusionMatrix[11]"] = cm[0][0]

    return stats