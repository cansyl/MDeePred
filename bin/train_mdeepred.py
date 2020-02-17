from __future__ import print_function, division
import os
import sys
import torch
import warnings
import numpy as np
import subprocess
from torch.autograd import Variable
from evaluation_metrics import pearson, spearman, ci, prec_rec_f1_acc_mcc, average_AUC, mse
from models import CompFCNNTarCNNModuleInception, CompFCNNTarCNN4Layers, CompFCNNTarCNNModule2Layers, CompFCNNTarCNN4LayersStride
from evaluation_metrics import r_squared_error, get_rm2, squared_error_zero, get_k, get_cindex, get_list_of_scores
from data_processing import get_cnn_test_val_folds_train_data_loader, get_cnn_train_test_full_training_data_loader, get_aa_match_encodings_max_value

warnings.filterwarnings(action='ignore')

cwd = os.getcwd()
# project_file_path = "{}PyTorch".format(cwd.split("PyTorch")[0])
training_files_path = "{}MDeePred/training_files".format(cwd.split("MDeePred")[0])
n_epoch = 100
num_of_folds = 5

def get_model(model_name, tar_feature_list, num_of_com_features, tar_num_of_last_neurons, comp_hidden_first, comp_hidden_second, fc1, fc2, dropout):
    model=None
    if model_name == "CompFCNNTarCNNModuleInception":
        model = CompFCNNTarCNNModuleInception(tar_feature_list, num_of_com_features, tar_num_of_last_neurons, comp_hidden_first, comp_hidden_second,
                                fc1, fc2, dropout)
    elif model_name=="CompFCNNTarCNN4Layers":
        model = CompFCNNTarCNN4Layers(tar_feature_list, num_of_com_features, tar_num_of_last_neurons, comp_hidden_first, comp_hidden_second,
                            fc1, fc2, dropout)
    elif model_name=="CompFCNNTarCNNModule2Layers":
        model = CompFCNNTarCNNModule2Layers(tar_feature_list, num_of_com_features, tar_num_of_last_neurons, comp_hidden_first, comp_hidden_second,
                            fc1, fc2, dropout)
    elif model_name=="CompFCNNTarCNN4LayersStride":
        model = CompFCNNTarCNN4LayersStride(tar_feature_list, num_of_com_features, tar_num_of_last_neurons, comp_hidden_first, comp_hidden_second,
                            fc1, fc2, dropout)
    return model


def get_scores(labels, predictions, validation_test, total_training_loss, total_validation_test_loss, epoch, fold_epoch_results, fold=None, print_scores=False):

    score_dict = {"rm2": None, "CI": None, "MSE": None, "Pearson": None,
                  "Spearman": None, "Average AUC": None,
                  "Precision 10uM": None, "Recall 10uM": None, "F1-Score 10uM": None, "Accuracy 10uM": None, "MCC 10uM": None,
                  "Precision 1uM": None, "Recall 1uM": None, "F1-Score 1uM": None, "Accuracy 1uM": None, "MCC 1uM": None,
                  "Precision 100nM": None, "Recall 100nM": None, "F1-Score 100nM": None, "Accuracy 100nM": None, "MCC 100nM": None,
                  "Precision 30nM": None, "Recall 30nM": None, "F1-Score 30nM": None, "Accuracy 30nM": None, "MCC 30nM": None,}
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

    if print_scores:
        if fold!=None:
            fold_epoch_results[-1].append(score_dict)
            print("Fold:{}\tEpoch:{}\tTraining Loss:{}\t{} Loss:{}".format(fold + 1, epoch, total_training_loss,
                                                                           validation_test, total_validation_test_loss))
        else:
            fold_epoch_results.append(score_dict)
            print("Epoch:{}\tTraining Loss:{}\t{} Loss:{}".format(epoch, total_training_loss, validation_test,
                                                                  total_validation_test_loss))
        for scr in score_list:
            print("{} {}:\t{}".format(validation_test, scr, score_dict[scr]))
    return score_dict

def five_fold_training(training_dataset, comp_feature_list, tar_feature_list, comp_hidden_lst, tar_num_of_last_neurons, fc1, fc2, learn_rate, batch_size, model_nm, dropout, experiment_name):
    arguments = [str(argm) for argm in
                 [training_dataset, comp_feature_list, tar_feature_list, comp_hidden_lst, tar_num_of_last_neurons, fc1, fc2, learn_rate, batch_size, model_nm, dropout, experiment_name]]
    print("Arguments:", "-".join(arguments))

    torch.manual_seed(123)
    np.random.seed(123)

    use_gpu = torch.cuda.is_available()

    device = "cpu"

    if use_gpu:
        print("GPU is available on this device!")
        device = "cuda"
    else:
        print("CPU is available on this device!")

    loader_fold_dict, test_loader = get_cnn_test_val_folds_train_data_loader(training_dataset, comp_feature_list, tar_feature_list, batch_size)

    validation_fold_epoch_results, test_fold_epoch_results = [], []

    if not os.path.exists("{}/result_files/{}".format(project_file_path, experiment_name)):
        subprocess.call("mkdir {}".format("{}/result_files/{}".format(project_file_path, experiment_name)),
                        shell=True)

    result_fl = open(
        "{}/result_files/{}/performance_results_{}.txt".format(project_file_path, experiment_name,  "-".join(arguments)), "w")
    prediction_fl = open(
        "{}/result_files/{}/predictions_{}.txt".format(project_file_path, experiment_name, "-".join(arguments)), "w")

    for fold in range(num_of_folds):

        best_performance_dict = dict()
        best_performance_dict["MSE"] = 100000000.0
        best_predictions = ""

        test_fold_epoch_results.append([])
        validation_fold_epoch_results.append([])
        train_loader, valid_loader = loader_fold_dict[fold]

        print("FOLD : {}".format(fold + 1))

        model = get_model(model_nm, tar_feature_list, 1024, tar_num_of_last_neurons, comp_hidden_lst[0],
                          comp_hidden_lst[1], fc1, fc2, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        criterion = torch.nn.MSELoss()
        optimizer.zero_grad()

        for epoch in range(n_epoch):
            print("Epoch :{}".format(epoch))
            total_training_loss, total_validation_loss, total_test_loss = 0.0, 0.0, 0.0
            total_training_count, total_validation_count, total_test_count = 0, 0, 0
            validation_predictions, validation_labels, test_predictions, test_labels = [], [], [], []
            test_all_comp_ids, test_all_tar_ids =  [], []
            batch_number = 0
            model.train()
            for i, data in enumerate(train_loader):
                batch_number += 1
                # clear gradient DO NOT forget you fool!
                optimizer.zero_grad()

                comp_feature_vectors, target_feature_vectors, labels, compound_ids, target_ids = data
                comp_feature_vectors, target_feature_vectors, labels = Variable(comp_feature_vectors).to(device), Variable(
                    target_feature_vectors).to(device), Variable(labels).to(device)

                total_training_count += comp_feature_vectors.shape[0]
                y_pred = model(comp_feature_vectors, target_feature_vectors).to(device)
                loss = criterion(y_pred.squeeze(), labels)
                total_training_loss += float(loss.item())
                loss.backward()
                optimizer.step()
            print("Epoch {} training loss:".format(epoch), total_training_loss)

            model.eval()
            with torch.no_grad():  # torch.set_grad_enabled(False):
                for i, data in enumerate(valid_loader):
                    val_comp_feature_vectors, val_target_feature_vectors, val_labels, val_compound_ids, val_target_ids = data
                    val_comp_feature_vectors, val_target_feature_vectors, val_labels = Variable(val_comp_feature_vectors).to(
                        device), Variable(
                        val_target_feature_vectors).to(device), Variable(val_labels).to(device)

                    total_validation_count += val_comp_feature_vectors.shape[0]
                    val_y_pred  = model(val_comp_feature_vectors, val_target_feature_vectors)
                    loss_val = criterion(val_y_pred.squeeze(), val_labels)
                    total_validation_loss += float(loss_val.item())
                    for item in val_labels:
                        validation_labels.append(float(item.item()))

                    for item in val_y_pred:
                        validation_predictions.append(float(item.item()))

                print("Epoch {} validation loss:".format(epoch), total_validation_loss)

                str_test_predictions = ""
                for i, data in enumerate(test_loader):
                    test_comp_feature_vectors, test_target_feature_vectors, tst_labels, test_compound_ids, test_target_ids = data
                    test_comp_feature_vectors, test_target_feature_vectors, tst_labels = Variable(test_comp_feature_vectors).to(
                        device), Variable(
                        test_target_feature_vectors).to(device), Variable(tst_labels).to(device)

                    total_test_count += test_comp_feature_vectors.shape[0]

                    test_y_pred  = model(test_comp_feature_vectors, test_target_feature_vectors)
                    loss_test = criterion(test_y_pred.squeeze(), tst_labels)
                    total_test_loss += float(loss_test.item())
                    for item in tst_labels:
                        test_labels.append(float(item.item()))

                    for item in test_y_pred:
                        test_predictions.append(float(item.item()))

                    test_all_comp_ids.extend(test_compound_ids)
                    test_all_tar_ids.extend(test_target_ids)
                # test_predictions, test_labels
                print_predictions = False
                if print_predictions:
                    print("=====PREDICTIONS=====")
                    for ind in range(len(test_all_tar_ids)):
                        print("{}\t{}\t{}\t{}".format(test_all_comp_ids[ind], test_all_tar_ids[ind], test_labels[ind],
                                                      test_predictions[ind]))

                    print("=====PREDICTIONS=====")
                for ind in range(len(test_all_tar_ids)):
                    str_test_predictions += "{}\t{}\t{}\t{}\n".format(test_all_comp_ids[ind], test_all_tar_ids[ind],
                                                                      test_labels[ind],
                                                                      test_predictions[ind])


            print("==============================================================================")
            validation_scores_dict = get_scores(validation_labels, validation_predictions, "Validation", total_training_loss, total_validation_loss, epoch, validation_fold_epoch_results, fold)
            print("------------------------------------------------------------------------------")
            test_scores_dict = get_scores(test_labels, test_predictions, "Test", total_training_loss,
                       total_test_loss, epoch, test_fold_epoch_results, fold)

            if test_scores_dict["MSE"]< best_performance_dict["MSE"]:
                print("OLD", best_performance_dict["MSE"])
                best_performance_dict = test_scores_dict
                best_predictions = str_test_predictions
                print("NEW", best_performance_dict["MSE"])



            if epoch==n_epoch-1:
                result_fl.write("FOLD : {}\n".format(fold + 1))
                score_list = get_list_of_scores()
                for scr in score_list:
                    result_fl.write("Test {}:\t{}\n".format(scr, best_performance_dict[scr]))

                prediction_fl.write("FOLD : {}\n".format(fold + 1))
                prediction_fl.write(best_predictions)



    result_fl.close()
    prediction_fl.close()

def train_val_test_training(training_dataset, comp_feature_list, tar_feature_list, comp_hidden_lst, tar_num_of_last_neurons, fc1, fc2, learn_rate, batch_size, model_nm, dropout, experiment_name):
    arguments = [str(argm) for argm in [training_dataset, comp_feature_list, tar_feature_list, comp_hidden_lst, tar_num_of_last_neurons, fc1, fc2, learn_rate, batch_size, model_nm, dropout, experiment_name]]
    print("Arguments:", "-".join(arguments))

    torch.manual_seed(123)
    np.random.seed(123)
    use_gpu = torch.cuda.is_available()

    device = "cpu"

    if use_gpu:
        print("GPU is available on this device!")
        device = "cuda"
    else:
        print("CPU is available on this device!")


    train_loader, validation_loader, test_loader = None, None, None

    train_loader, validation_loader, test_loader = get_cnn_train_test_full_training_data_loader(training_dataset, comp_feature_list, tar_feature_list, batch_size, True)


    if not os.path.exists("{}/result_files/{}".format(project_file_path, experiment_name)):
        subprocess.call("mkdir {}".format("{}/result_files/{}".format(project_file_path, experiment_name)),
                        shell=True)

    result_fl = open(
        "{}/result_files/{}/{}.txt".format(project_file_path, experiment_name,  "-".join(arguments)), "w")
    prediction_fl = open(
        "{}/result_files/{}/predictions_{}.txt".format(project_file_path, experiment_name, "-".join(arguments)), "w")

    best_performance_dict = dict()
    best_performance_dict["MSE"] = 100000000.0
    best_predictions = ""

    validation_epoch_results, test_epoch_results = [], []
    validation_epoch_results.append([])
    test_epoch_results.append([])

    model = get_model(model_nm, tar_feature_list, 1024, tar_num_of_last_neurons, comp_hidden_lst[0], comp_hidden_lst[1], fc1, fc2, dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = torch.nn.MSELoss()
    optimizer.zero_grad()

    for epoch in range(n_epoch):
        print("Epoch :{}".format(epoch))
        total_training_loss, total_test_loss, total_validation_loss = 0.0, 0.0, 0.0
        total_training_count, total_test_count, total_validation_count = 0, 0, 0
        test_predictions, test_labels, test_all_comp_ids, test_all_tar_ids = [], [], [], []
        validation_predictions, validation_labels, validation_all_comp_ids, validation_all_tar_ids = [], [], [], []

        batch_number = 0

        model.train()
        for i, data in enumerate(train_loader):
            batch_number += 1
            # clear gradient DO NOT forget you fool!
            optimizer.zero_grad()

            # get the inputs
            comp_feature_vectors, target_feature_vectors, labels, compound_ids, target_ids = data
            comp_feature_vectors, target_feature_vectors, labels = Variable(comp_feature_vectors).to(device), Variable(
                target_feature_vectors).to(device), Variable(labels).to(device)

            total_training_count += comp_feature_vectors.shape[0]
            y_pred = model(comp_feature_vectors, target_feature_vectors).to(device)
            loss = criterion(y_pred.squeeze(), labels)
            total_training_loss += float(loss.item())
            loss.backward()
            optimizer.step()

        print("Epoch {} training loss:".format(epoch), total_training_loss)

        model.eval()
        with torch.no_grad():  # torch.set_grad_enabled(False):

            for i, data in enumerate(validation_loader):
                validation_comp_feature_vectors, validation_target_feature_vectors, val_labels, validation_compound_ids, validation_target_ids = data
                validation_comp_feature_vectors, validation_target_feature_vectors, val_labels = Variable(
                    validation_comp_feature_vectors).to(
                    device), Variable(
                    validation_target_feature_vectors).to(device), Variable(val_labels).to(device)

                total_validation_count += validation_comp_feature_vectors.shape[0]

                validation_y_pred = model(validation_comp_feature_vectors, validation_target_feature_vectors)
                loss_validation = criterion(validation_y_pred.squeeze(), val_labels)
                total_validation_loss += float(loss_validation.item())

                for item in val_labels:
                    validation_labels.append(float(item.item()))

                for item in validation_y_pred:
                    validation_predictions.append(float(item.item()))

                for item in validation_compound_ids:
                    validation_all_comp_ids.append(item)

                for item in validation_target_ids:
                    validation_all_tar_ids.append(item)


        print("==============================================================================")
        validation_scores_dict = get_scores(validation_labels, validation_predictions, "validation", total_training_loss,
                        total_validation_loss, epoch, validation_epoch_results)

        print("Epoch {} validation loss:".format(epoch), total_validation_loss)
        str_test_predictions =  ""
        model.eval()
        with torch.no_grad():  # torch.set_grad_enabled(False):

            for i, data in enumerate(test_loader):
                test_comp_feature_vectors, test_target_feature_vectors, tst_labels, test_compound_ids, test_target_ids = data
                test_comp_feature_vectors, test_target_feature_vectors, tst_labels = Variable(test_comp_feature_vectors).to(
                    device), Variable(
                    test_target_feature_vectors).to(device), Variable(tst_labels).to(device)

                total_test_count += test_comp_feature_vectors.shape[0]

                test_y_pred = None
                test_y_pred  = model(test_comp_feature_vectors, test_target_feature_vectors)
                loss_test = criterion(test_y_pred.squeeze(), tst_labels)
                total_test_loss += float(loss_test.item())
                for item in tst_labels:
                    test_labels.append(float(item.item()))

                for item in test_y_pred:
                    test_predictions.append(float(item.item()))

                for item in test_compound_ids:
                    test_all_comp_ids.append(item)

                for item in test_target_ids:
                    test_all_tar_ids.append(item)
            # print(test_all_tar_ids)
            print_predictions = False
            if print_predictions:
                print("=====PREDICTIONS=====")
                for ind in range(len(test_all_tar_ids)):
                    print("{}\t{}\t{}\t{}".format(test_all_comp_ids[ind], test_all_tar_ids[ind], test_labels[ind], test_predictions[ind]))
                print("=====PREDICTIONS=====")

            for ind in range(len(test_all_tar_ids)):
                str_test_predictions += ("{}\t{}\t{}\t{}".format(test_all_comp_ids[ind], test_all_tar_ids[ind], test_labels[ind],
                                              test_predictions[ind]))


        print("==============================================================================")
        test_scores_dict = get_scores(test_labels, test_predictions, "Test", total_training_loss,
                   total_test_loss, epoch, test_epoch_results)

        if test_scores_dict["MSE"] < best_performance_dict["MSE"]:
            best_performance_dict = test_scores_dict
            best_predictions = str_test_predictions


        if epoch == n_epoch - 1:
            result_fl.write("FOLD : {}\n".format(fold + 1))
            score_list = get_list_of_scores()
            for scr in score_list:
                result_fl.write("Test {}:\t{}\n".format(scr, best_performance_dict[scr]))

            prediction_fl.write("FOLD : {}\n".format(fold + 1))
            prediction_fl.write(best_predictions)

    result_fl.close()
    prediction_fl.close()



