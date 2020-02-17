import numpy as np
import copy
from math import sqrt
from scipy import stats
from sklearn import preprocessing, metrics

def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair is not 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


"""
@author: Anna Cichonska
"""



def mse(y,f):
    """
    Task:    To compute root mean squared error (RMSE)

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  mse   MSE
    """

    mse = ((y - f)**2).mean(axis=0)

    return mse


def pearson(y,f):
    """
    Task:    To compute Pearson correlation coefficient

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rp     Pearson correlation coefficient
    """

    rp = np.corrcoef(y, f)[0,1]

    return rp


def spearman(y,f):
    """
    Task:    To compute Spearman's rank correlation coefficient

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rs     Spearman's rank correlation coefficient
    """

    rs = stats.spearmanr(y, f)[0]

    return rs



def prec_rec_f1_acc_mcc(y,f):
    """
    Task:    To compute F1 score using the threshold of 7 M
             to binarize pKd's into true class labels.

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  f1     F1 score
    """
    # 10 uM, 1 uM, 100 nM
    str_threshold_lst = ["10uM", "1uM", "100nM", "30nM"]
    threshold_lst = [5.0, 6.0, 7.0, 7.522878745280337]
    dict_threshold = {str_threshold_lst[0]:threshold_lst[0] ,str_threshold_lst[1]:threshold_lst[1],
                      str_threshold_lst[2]:threshold_lst[2], str_threshold_lst[3]:threshold_lst[3]}

    performance_threshold_dict = dict()
    for str_thre, threshold in dict_threshold.items():
        y_binary = copy.deepcopy(y)
        y_binary = preprocessing.binarize(y_binary.reshape(1,-1), threshold, copy=False)[0]
        # print(y_binary)
        f_binary = copy.deepcopy(f)
        f_binary = preprocessing.binarize(f_binary.reshape(1,-1), threshold, copy=False)[0]
        precision = metrics.precision_score(y_binary, f_binary)
        recall = metrics.recall_score(y_binary, f_binary)
        f1_score = metrics.f1_score(y_binary, f_binary)
        accuracy = metrics.accuracy_score(y_binary, f_binary)
        mcc = metrics.matthews_corrcoef(y_binary, f_binary)
        performance_threshold_dict["Precision {}".format(str_thre)] = precision
        performance_threshold_dict["Recall {}".format(str_thre)] = recall
        performance_threshold_dict["F1-Score {}".format(str_thre)] = f1_score
        performance_threshold_dict["Accuracy {}".format(str_thre)] = accuracy
        performance_threshold_dict["MCC {}".format(str_thre)] = mcc

    return performance_threshold_dict


def average_AUC(y,f):

    """
    Task:    To compute average area under the ROC curves (AUC) given ten
             interaction threshold values from the pKd interval [6 M, 8 M]
             to binarize pKd's into true class labels.

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  avAUC   average AUC

    """

    thr = np.linspace(6,8,10)
    auc = np.empty(np.shape(thr)); auc[:] = np.nan

    for i in range(len(thr)):
        y_binary = copy.deepcopy(y)
        y_binary = preprocessing.binarize(y_binary.reshape(1,-1), threshold=thr[i], copy=False)[0]
        fpr, tpr, thresholds = metrics.roc_curve(y_binary, f, pos_label=1)
        auc[i] = metrics.auc(fpr, tpr)

    avAUC = np.mean(auc)

    return avAUC


def get_list_of_scores():
    score_list = ["rm2", "CI", "MSE", "Pearson", "Spearman",
                  "Average AUC",
                  "Precision 10uM", "Recall 10uM", "F1-Score 10uM", "Accuracy 10uM", "MCC 10uM",
                  "Precision 1uM", "Recall 1uM", "F1-Score 1uM", "Accuracy 1uM", "MCC 1uM",
                  "Precision 100nM", "Recall 100nM", "F1-Score 100nM", "Accuracy 100nM", "MCC 100nM",
                  "Precision 30nM", "Recall 30nM", "F1-Score 30nM", "Accuracy 30nM", "MCC 30nM",
                  ]
    return score_list


def get_validation_test_metric_list_of_scores():
    score_list =  get_list_of_scores()
    test_score_list = ["test {}".format(scr) for scr in score_list]
    validation_score_list = ["validation {}".format(scr) for scr in score_list]
    validation_test_metric_list = test_score_list + validation_score_list
    # print(validation_test_list)
    return validation_test_metric_list

def get_scores_generic(labels, predictions, validation_test, single_line_print=False):
    score_dict = {"rm2": None, "CI": None, "MSE": None, "Pearson": None,
                  "Spearman": None,  "Average AUC": None,
                  "Precision 10uM": None, "Recall 10uM": None, "F1-Score 10uM": None, "Accuracy 10uM": None,
                  "MCC 10uM": None,
                  "Precision 1uM": None, "Recall 1uM": None, "F1-Score 1uM": None, "Accuracy 1uM": None,
                  "MCC 1uM": None,
                  "Precision 100nM": None, "Recall 100nM": None, "F1-Score 100nM": None, "Accuracy 100nM": None,
                  "MCC 100nM": None,
                  "Precision 30nM": None, "Recall 30nM": None, "F1-Score 30nM": None, "Accuracy 30nM": None,
                  "MCC 30nM": None, }
    score_list = get_list_of_scores()

    score_dict["rm2"] = get_rm2(np.asarray(labels), np.asarray(
        predictions))
    score_dict["CI"] = get_cindex(np.asarray(labels), np.asarray(
        predictions))
    score_dict["MSE"] = mse(np.asarray(labels), np.asarray(
        predictions))
    score_dict["Pearson"] = pearson(np.asarray(labels), np.asarray(predictions))
    score_dict["Spearman"] = spearman(np.asarray(labels), np.asarray(predictions))
    score_dict["Average AUC"] = average_AUC(np.asarray(labels), np.asarray(predictions))

    prec_rec_f1_acc_mcc_threshold_dict = prec_rec_f1_acc_mcc(np.asarray(labels), np.asarray(predictions))
    for key in prec_rec_f1_acc_mcc_threshold_dict.keys():
        score_dict[key] = prec_rec_f1_acc_mcc_threshold_dict[key]


    str_single_line_performances = ""
    if single_line_print:
        print("\t".join(score_list))
        for scr in score_list:
            str_single_line_performances += "{}\t".format(score_dict[scr])
        print(str_single_line_performances)
    else:
        for scr in score_list:
            print("{} {}:\t{}".format(validation_test, scr, score_dict[scr]))

