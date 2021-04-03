#!/usr/bin/python
import numpy as np

file = np.loadtxt('HW2_labels.txt',  delimiter=',')
y_predict, y_true = file[:, :2], file[:, -1]

y_predict = np.asarray(y_predict)
y_true = np.asarray(y_true)

def get_pred(y_true, y_predict, percent=None):
    tmp =  np.vstack((y_predict[:,1], y_true)).T
    tmp = np.flipud(tmp[tmp[:,0].argsort()])
    threshold = 0.5 if percent == None else tmp[int(percent*tmp.shape[0]/100) -1, 0]
    return np.array([1 if tmp[i,0] > threshold else 0 for i in range(tmp.shape[0])])

def accuracy_score(y_true, y_predict, percent=None):
    y_pred = get_pred(y_true, y_predict, percent)
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_predict, percent=None):
    y_pred = get_pred(y_true, y_predict, percent)
    TP = 0
    FP = 0
    for i in range(y_pred.shape[0]):
        if (y_pred[i] == 1):
            if (y_true[i] == 1):
                TP += 1
            else:
                FP += 1
    return TP/(TP+FP)


def recall_score(y_true, y_predict, percent=None):
    y_pred = get_pred(y_true, y_predict, percent)
    TP = 0
    FN = 0
    for i in range(y_pred.shape[0]):
        if (y_true[i] == 1):
            if (y_pred[i] == 1):
                TP += 1
            else:
                FN += 1
    return TP/(TP+FN)

def lift_score(y_true, y_predict, percent=None):
    y_pred = get_pred(y_true, y_predict, percent)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(y_pred.shape[0]):
        if (y_true[i] == 1):
            if (y_pred[i] == 1):
                TP += 1
            else:
                FN += 1
        else:
            if (y_pred[i] == 1):
                FP += 1
            else:
                TN += 1
    return (TP/(TP+FP))/((TP+FN)/(TP+TN+FP+FN))

def f1_score(y_true, y_predict, percent=None):
    precision = precision_score(y_true, y_predict, percent)
    recall = recall_score(y_true, y_predict, percent)
    return 2 * (precision * recall) / (precision + recall)