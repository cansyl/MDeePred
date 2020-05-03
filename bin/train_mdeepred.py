from __future__ import print_function, division
import os
import sys
import torch
import warnings
import numpy as np
import subprocess
from torch.autograd import Variable
from models import CompFCNNTarCNNModuleInception, CompFCNNTarCNN4Layers, CompFCNNTarCNNModule2Layers, CompFCNNTarCNN4LayersStride, get_model
from evaluation_metrics import pearson, spearman, get_cindex, prec_rec_f1_acc_mcc, average_AUC, average_AUPRC, mse
from evaluation_metrics import r_squared_error, get_rm2, squared_error_zero, get_k, get_cindex, get_list_of_scores, get_scores
from data_processing import get_cnn_test_val_folds_train_data_loader, get_cnn_train_test_full_training_data_loader, get_aa_match_encodings_max_value

warnings.filterwarnings(action='ignore')

cwd = os.getcwd()
project_file_path = "{}MDeePred".format(cwd.split("MDeePred")[0])
training_files_path = "{}MDeePred/training_files".format(cwd.split("MDeePred")[0])

def compute_test_loss(model, criterion, data_loader, device):
    total_count = 0
    total_loss = 0.0
    all_comp_ids = []
    all_tar_ids = []
    all_labels = []
    predictions = []
    for i, data in enumerate(data_loader):
        comp_feature_vectors, target_feature_vectors, labels, compound_ids, target_ids = data
        comp_feature_vectors, target_feature_vectors, labels = Variable(comp_feature_vectors).to(
            device), Variable(
            target_feature_vectors).to(device), Variable(labels).to(device)
        all_comp_ids.extend(compound_ids)
        all_tar_ids.extend(target_ids)
        total_count += comp_feature_vectors.shape[0]
        y_pred = model(comp_feature_vectors, target_feature_vectors)
        loss_val = criterion(y_pred.squeeze(), labels)
        total_loss += float(loss_val.item())
        for item in labels:
            all_labels.append(float(item.item()))

        for item in y_pred:
            predictions.append(float(item.item()))

    return total_loss, total_count, all_labels, predictions, all_comp_ids, all_tar_ids

def save_best_model_predictions(test_validation, epoch, test_scores_dict, old_mse_score, new_mse_score, model, project_file_path, training_dataset, str_arguments,
                                                                                   all_test_comp_ids, all_test_tar_ids, test_labels, test_predictions, fold=None):
    # print("Model ({}) is being saved.\tEpoch:{}\tOld MSE:{}\tNew MSE:{}".format(test_validation, epoch, old_mse_score,
    #                                                                                    new_mse_score))
    best_val_fold_mse_score = new_mse_score
    torch.save(model.state_dict(),
               "{}/trained_models/{}/{}_best_val_{}_fold_{}_state_dict.pth".format(project_file_path, training_dataset,
                                                                                   training_dataset, str_arguments,
                                                                                   fold))
    str_test_predictions = ""
    for ind in range(len(all_test_tar_ids)):
        str_test_predictions += "{}\t{}\t{}\t{}\n".format(all_test_comp_ids[ind], all_test_tar_ids[ind],
                                                          test_labels[ind],
                                                          test_predictions[ind])
    best_test_performance_dict = test_scores_dict
    best_test_predictions = str_test_predictions
    return best_test_performance_dict, best_test_predictions, best_val_fold_mse_score, str_test_predictions


def five_fold_training(training_dataset, comp_feature_list, tar_feature_list, comp_hidden_lst, tar_num_of_last_neurons, fc1, fc2, learn_rate, batch_size, model_nm, dropout, experiment_name, n_epoch, fold_num=None, external_comp_feat_fl=None):
    arguments = [str(argm) for argm in [comp_hidden_lst[0], comp_hidden_lst[1], tar_num_of_last_neurons, fc1, fc2, learn_rate, batch_size, model_nm, dropout, n_epoch, fold_num]]

    str_arguments = "-".join(arguments)
    print("Arguments:", str_arguments)

    torch.manual_seed(123)
    np.random.seed(123)

    use_gpu = torch.cuda.is_available()

    device = "cpu"

    if use_gpu:
        print("GPU is available on this device!")
        device = "cuda"
    else:
        print("CPU is available on this device!")

    #get_cnn_test_val_folds_train_data_loader("Davis_Filtered", ["ecfp4"], ["sequencematrix500"], external_comp_feat_fl="aacrtest_ecfp4_normalized.tsv")
    loader_fold_dict, test_loader, external_data_loader = get_cnn_test_val_folds_train_data_loader(training_dataset,
                                                                                                   comp_feature_list,
                                                                                                   tar_feature_list,
                                                                                                   batch_size,
                                                                                                   external_comp_feat_fl)
    num_of_folds = len(loader_fold_dict)
    validation_fold_epoch_results, test_fold_epoch_results = [], []

    if not os.path.exists("{}/result_files/{}".format(project_file_path, experiment_name)):
        subprocess.call("mkdir {}".format("{}/result_files/{}".format(project_file_path, experiment_name)),
                        shell=True)

    best_test_result_fl = open(
        "{}/result_files/{}/test_performance_results-{}.txt".format(project_file_path, experiment_name,
                                                                             "-".join(arguments)), "w")

    best_test_prediction_fl = open(
        "{}/result_files/{}/test_predictions-{}.txt".format(project_file_path, experiment_name,
                                                                      "-".join(arguments)), "w")

    folds = range(num_of_folds) if not fold_num else range(fold_num, fold_num + 1)
    print(list(folds))
    for fold in folds:
        best_val_fold_mse_score, best_test_fold_mse_score = 10000, 10000
        best_val_test_performance_dict, best_test_test_performance_dict = dict(), dict()
        best_val_test_performance_dict["MSE"], best_test_test_performance_dict["MSE"] = 100000000.0, 100000000.0

        str_best_val_test_predictions = ""
        str_best_test_test_predictions = ""

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
            # validation_predictions, validation_labels, test_predictions, test_labels = [], [], [], []
            # test_all_comp_ids, test_all_tar_ids =  [], []
            batch_number = 0
            model.train()
            print("Training:", model.training)
            for i, data in enumerate(train_loader):
                batch_number += 1
                # clear gradient DO NOT forget you fool!
                optimizer.zero_grad()

                comp_feature_vectors, target_feature_vectors, labels, compound_ids, target_ids = data
                comp_feature_vectors, target_feature_vectors, labels = Variable(comp_feature_vectors).to(
                    device), Variable(
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
                print("Training:", model.training)
                total_validation_loss, total_validation_count, validation_labels, validation_predictions, all_val_comp_ids, all_val_tar_ids = compute_test_loss(
                    model, criterion, valid_loader, device)
                print("Epoch {} validation loss:".format(epoch), total_validation_loss)

                total_test_loss, total_test_count, test_labels, test_predictions, all_test_comp_ids, all_test_tar_ids = compute_test_loss(
                    model, criterion, test_loader, device)

                print("==============================================================================")
                validation_scores_dict = get_scores(validation_labels, validation_predictions, "Validation",
                                                    total_training_loss, total_validation_loss, epoch,
                                                    validation_fold_epoch_results, False, fold)

                print("------------------------------------------------------------------------------")
                test_scores_dict = get_scores(test_labels, test_predictions, "Test", total_training_loss,
                                              total_test_loss, epoch, test_fold_epoch_results, False, fold)

                if test_scores_dict["MSE"] < best_test_fold_mse_score:
                    best_test_test_performance_dict, best_test_test_predictions, best_test_fold_mse_score, str_best_test_test_predictions = save_best_model_predictions(
                        "Test", epoch, test_scores_dict, best_test_fold_mse_score, test_scores_dict["MSE"], model,
                        project_file_path, training_dataset, str_arguments,
                        all_test_comp_ids, all_test_tar_ids, test_labels, test_predictions, fold)

            if epoch == n_epoch - 1:
                best_test_prediction_fl.write("FOLD : {}\n".format(fold + 1))
                best_test_result_fl.write("FOLD : {}\n".format(fold + 1))
                score_list = get_list_of_scores()
                for scr in score_list:
                    best_test_result_fl.write("Test {}:\t{}\n".format(scr, best_test_test_performance_dict[scr]))

                best_test_prediction_fl.write("FOLD : {}\n".format(fold + 1))
                best_test_prediction_fl.write(best_test_test_predictions)

    best_test_prediction_fl.close()
    best_test_result_fl.close()


def train_val_test_training(training_dataset, comp_feature_list, tar_feature_list, comp_hidden_lst, tar_num_of_last_neurons, fc1, fc2, learn_rate, batch_size, model_nm, dropout, experiment_name, n_epoch, train_val_test=True, external_comp_feat_fl=None):
    arguments = [str(argm) for argm in [comp_hidden_lst[0], comp_hidden_lst[1], tar_num_of_last_neurons, fc1, fc2, learn_rate, batch_size, model_nm, dropout, n_epoch, train_val_test, external_comp_feat_fl]]
    str_arguments = "-".join(arguments)
    print("Arguments:", str_arguments)

    torch.manual_seed(123)
    np.random.seed(123)
    use_gpu = torch.cuda.is_available()

    device = "cpu"

    if use_gpu:
        print("GPU is available on this device!")
        device = "cuda"
    else:
        print("CPU is available on this device!")


    train_loader, valid_loader, test_loader, external_test_loader = None, None, None, None
    best_val_test_predictions, best_test_test_predictions = None, None
    if train_val_test:
        train_loader, valid_loader, test_loader, external_test_loader = get_cnn_train_test_full_training_data_loader(training_dataset, comp_feature_list, tar_feature_list, batch_size, train_val_test, external_comp_feat_fl)
    else:
        train_loader, test_loader, external_test_loader = get_cnn_train_test_full_training_data_loader(
            training_dataset, comp_feature_list, tar_feature_list, batch_size, train_val_test, external_comp_feat_fl)
    if not os.path.exists("{}/result_files/{}".format(project_file_path, experiment_name)):
        subprocess.call("mkdir {}".format("{}/result_files/{}".format(project_file_path, experiment_name)),
                        shell=True)

    best_val_test_result_fl, best_val_test_prediction_fl = None, None
    if train_val_test:
        best_val_test_result_fl = open(
            "{}/result_files/{}/best_val_test_performance_results-{}.txt".format(project_file_path, experiment_name,  "-".join(arguments)), "w")
        best_val_test_prediction_fl = open(
            "{}/result_files/{}/best_val_test_predictions-{}.txt".format(project_file_path, experiment_name,
                                                                         "-".join(arguments)), "w")

    best_test_test_result_fl = open(
        "{}/result_files/{}/best_test_test_performance_results-{}.txt".format(project_file_path, experiment_name,
                                                                            "-".join(arguments)), "w")

    best_test_test_prediction_fl = open(
        "{}/result_files/{}/best_test_test_predictions-{}.txt".format(project_file_path, experiment_name,
                                                                     "-".join(arguments)), "w")

    best_val_mse_score, best_test_mse_score = 10000, 10000
    best_val_test_performance_dict, best_test_test_performance_dict = dict(), dict()
    best_val_test_performance_dict["MSE"], best_test_test_performance_dict["MSE"] = 100000000.0, 100000000.0

    test_epoch_results = []
    validation_epoch_results = []

    model = get_model(model_nm, tar_feature_list, 1024, tar_num_of_last_neurons, comp_hidden_lst[0],
                      comp_hidden_lst[1], fc1, fc2, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = torch.nn.MSELoss()
    optimizer.zero_grad()

    for epoch in range(n_epoch):
        print("Epoch :{}".format(epoch))
        total_training_loss, total_validation_loss, total_test_loss = 0.0, 0.0, 0.0
        total_training_count, total_validation_count, total_test_count = 0, 0, 0
        # validation_predictions, validation_labels, test_predictions, test_labels = [], [], [], []
        # test_all_comp_ids, test_all_tar_ids =  [], []
        batch_number = 0
        model.train()
        print("Training:", model.training)
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
            if train_val_test:
                total_validation_loss, total_validation_count, validation_labels, validation_predictions, all_val_comp_ids, all_val_tar_ids = compute_test_loss(
                    model, criterion, valid_loader, device)
                print("Epoch {} validation loss:".format(epoch), total_validation_loss)
                print("==============================================================================")
                validation_scores_dict = get_scores(validation_labels, validation_predictions, "Validation",
                                                    total_training_loss, total_validation_loss, epoch,
                                                    validation_epoch_results)
                if validation_scores_dict["MSE"] < best_val_mse_score:
                    best_val_test_performance_dict, best_val_test_predictions, best_val_mse_score, str_best_val_test_predictions = save_best_model_predictions(
                        "Validation", epoch, test_scores_dict, best_val_mse_score, validation_scores_dict["MSE"],
                        model,
                        project_file_path, training_dataset, str_arguments,
                        all_test_comp_ids, all_test_tar_ids, test_labels, test_predictions)


            total_test_loss, total_test_count, test_labels, test_predictions, all_test_comp_ids, all_test_tar_ids = compute_test_loss(
                model, criterion, test_loader, device)
            print("Epoch {} test loss:".format(epoch), total_test_loss)
            print("------------------------------------------------------------------------------")
            test_scores_dict = get_scores(test_labels, test_predictions, "Test", total_training_loss,
                                          total_test_loss, epoch, test_epoch_results)

            if test_scores_dict["MSE"] < best_test_mse_score:

                best_test_test_performance_dict, best_test_test_predictions, best_test_mse_score, str_best_test_test_predictions = save_best_model_predictions(
                    "Test", epoch, test_scores_dict, best_test_mse_score, test_scores_dict["MSE"], model,
                    project_file_path, training_dataset, str_arguments,
                    all_test_comp_ids, all_test_tar_ids, test_labels, test_predictions)


        if epoch == n_epoch - 1:
            score_list = get_list_of_scores()
            if train_val_test:
                for scr in score_list:
                    best_val_test_result_fl.write("Test {}:\t{}\n".format(scr, best_val_test_performance_dict[scr]))
                best_val_test_prediction_fl.write(best_val_test_predictions)

            for scr in score_list:
                best_test_test_result_fl.write("Test {}:\t{}\n".format(scr, best_test_test_performance_dict[scr]))
            best_test_test_prediction_fl.write(best_test_test_predictions)


    if train_val_test:
        best_val_test_result_fl.close()
        best_val_test_prediction_fl.close()
    best_test_test_result_fl.close()
    best_test_test_prediction_fl.close()