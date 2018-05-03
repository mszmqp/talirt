# coding=utf-8

'''
Created on 2018年1月27日
@author: mengshuai@100tal.com
function: get_mse
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def get_mse(pred, actual):
    # get the subset of the two sets.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

def multy_key_in(result, test):
    '''
    :param result:
    :param test:
    :output: the six index of the algorithm
    tpr/fpr,precision/recall/accuracy
    auc
    '''
    from sklearn import metrics
    pred = result[test.nonzero()].flatten()
    # transform the 1-2 to 0-1
    pred = (pred+1)/2
    for i in range(len(pred)):
        pred[i] = round(pred[i], 1)
    actual = test[test.nonzero()].flatten()
    actual = (actual+1)/2

    fpr, tpr, thresholds = metrics.roc_curve(actual, pred, pos_label=1)

    pred = np.asmatrix(pred)
    actual = np.asmatrix(actual)
    max_acc = 0.5
    for i in range(len(thresholds)):
        if thresholds[i] < 0.3 or thresholds[i] > 0.6:
            continue
        pre_temp = pred.copy()
        # print("the thresholds is %s" % thresholds[i])
        # print("the tpr is %s" % tpr[i])
        # print("the fpr is %s" % fpr[i])
        if pred[pred >= thresholds[i]].shape[1] == 0:
            pre = 0
        else:
            pre = tpr[i] * actual[actual >= thresholds[i]].shape[1] / pred[pred >= thresholds[i]].shape[1]
        rec = tpr[i]
        pre_temp[pred >= thresholds[i]] = 1
        pre_temp[pred < thresholds[i]] = 0
        dif = pre_temp - actual
        acc = float(dif[dif == 0].shape[1]) / dif.shape[1]
        if acc > max_acc:
            max_acc = acc
        # print("the precision is %s" % pre)
        # print("the recall is %s" % rec)
        # print("the accuracy is %s" % acc)

    auc = metrics.auc(fpr, tpr)
    print('The auc is %0.2f' % auc)
    print('The max_acc is %0.5f' % max_acc)
    print('\n')
    return max_acc

    '''
    plt.switch_backend('agg')
    # 画出roc曲线
    plt.title('Receiver Operating Characteristic')\

    plt.plot(fpr, tpr, 'b',
             label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    '''
