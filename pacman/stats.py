__author__ = 'milan'

# Inbuilt libraries
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

# analyze function generate stats regarding the prediction and returns all the stats in the form
# of a pandas dataframe
# Input:
#   actual - ground truth energy consumption
#   predicted - predicted energy consumption
#   room - room number
#   usage - AC usage number
#   model_type - type of thermal model
#   regressor - machine learning algorithm used to learn the thermal model
#   classifier - machine learning algorithm used to predict AC compressor state
#   count - 
#   pusages - size of training dataset
#   usage_act - actual id of the AC usage
#   Tset - AC set temperature
#   prated - AC's rated power consumption
#   manufacturer - AC's manufacturer 
# Output:
#   stats - output frame having evaluation of prediction on various metrics
def analyze(actual, predicted, room, usage, model_type, regressor, classifier, count, pusages, usage_act, Tset,
            prated, manufacturer):
    
    # Initialise a stats frame
    stats = pd.DataFrame()

    # Actual and predicted energy consumption of the AC
    E_act = actual.sum()
    E_est = predicted.sum()

    # Deviation, absolute deviation, and accuracy numbers for the prediction
    error = E_act - E_est
    error_abs = np.abs(error)
    acc = 100 - (error_abs*100.0)/(E_act)
    
    # Precision, recall, fscore, and confusion matrix to evaluate the accuracy in 
    # predicting AC compressor state
    prfs = precision_recall_fscore_support(actual.values, predicted.values)
    cm = confusion_matrix(actual.values, predicted.values)

    # Prediction count
    approach = count

    # Write the stats
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

    # Check to avoid null values
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

    # Return the stats frame
    return stats


# analyze_est function generate stats regarding the estimation and returns all the stats in the form
# of a pandas dataframe
# Input:
#   actual - ground truth energy consumption
#   predicted - predicted energy consumption
#   room - room number
#   usage - AC usage number
#   usage_act - actual id of the AC usage
#   Tset - AC set temperature
#   prated - AC's rated power consumption
#   manufacturer - AC's manufacturer 
# Output:
#   stats - output frame having evaluation of estimation on various metrics
def analyze_est(actual, predicted, room, usage, usage_act, Tset, prated, manufacturer):

    # Initialise a stats frame
    stats = pd.DataFrame()

    # Actual and estimated energy consumption of the AC
    E_act = actual.sum()
    E_est = predicted.sum()

    # Deviation, absolute deviation, and accuracy numbers for the prediction
    error = E_act - E_est
    error_abs = np.abs(error)
    acc = 100 - (error_abs*100.0)/(E_act)

    # Precision, recall, fscore, and confusion matrix to evaluate the accuracy in 
    # predicting AC compressor state
    prfs = precision_recall_fscore_support(actual.values, predicted.values)
    cm = confusion_matrix(actual.values, predicted.values)

    # Estimation count
    approach = 1

    # Write the stats
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

    # Check to avoid null values
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

    # Return the stats frame
    return stats