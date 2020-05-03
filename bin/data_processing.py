import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler, SequentialSampler
import os


cwd = os.getcwd()
#training_files_path = "{}PyTorch/trainingFiles".format(cwd.split("PyTorch")[0])
training_files_path = "{}MDeePred/training_files".format(cwd.split("MDeePred")[0])
compound_target_pair_dataset = "comp_targ_affinity.csv"


def get_aa_match_encodings_max_value(aaindex_enconding):
    import math
    encoding_fl = open("{}/encodings/{}.txt".format(training_files_path, aaindex_enconding))
    lst_encoding_fl = encoding_fl.read().split("\n")
    encoding_fl.close()
    starting_ind = -1
    max_value = -1000000000
    for row_ind in range(len(lst_encoding_fl)-1):
        str_line = lst_encoding_fl[row_ind]
        if str_line.startswith("M rows"):
            starting_ind = row_ind + 1

        if  not str_line.startswith("//") and starting_ind != -1 and row_ind >= starting_ind:
            str_line = str_line.split(" ")

            while "" in str_line:
                str_line.remove("")

            for col_ind in range(len(str_line)):
                max_value = max(max_value, round(float(str_line[col_ind]),3))

    return max_value

# print(get_aa_match_encodings_max_value("SIMK990101.txt"))

def get_max_values_for_target_types(tar_feature_list):
    tar_feat_max_dict = dict()
    tar_feat_max_dict["sequencematrix500"] = 210.0
    tar_feat_max_dict["sequencematrix1000"] = 210.0
    for tar_feat in tar_feature_list[1:]:
        tar_feat = tar_feat.split("LEQ")[0]
        tar_feat_max_dict[tar_feat] = get_aa_match_encodings_max_value(tar_feat)

    return tar_feat_max_dict


def get_chembl_target_id_uniprot_mapping():
    chembl_uniprot_dict = dict()
    chembl_training_files_path = "{}/ChEMBL25/helper_files".format(training_files_path)
    with open("{}/{}".format(chembl_training_files_path, "chembl_uniprot_mapping.txt")) as f:
        for line in f:
            if not line.startswith("#") and line != "":
                line=line.split("\n")[0]
                u_id, chembl_id, defin, target_type = line.split("\t")

                if target_type=='SINGLE PROTEIN':

                    try:
                        chembl_uniprot_dict[chembl_id].append(u_id)
                    except:
                        chembl_uniprot_dict[chembl_id] = [u_id]

    for key in chembl_uniprot_dict.keys():
        if len(chembl_uniprot_dict[key])!=1:
            print(key, chembl_uniprot_dict[key])
    return chembl_uniprot_dict

#get_chembl_target_id_uniprot_mapping()


def get_chembl_target_id_protein_name_mapping():
    chembl_def_dict = dict()
    chembl_training_files_path = "{}/ChEMBL25/helper_files".format(training_files_path)
    with open("{}/{}".format(chembl_training_files_path, "chembl_uniprot_mapping.txt")) as f:
        for line in f:
            if not line.startswith("#") and line != "":
                line=line.split("\n")[0]
                u_id, chembl_id, defin, target_type = line.split("\t")
                if target_type=='SINGLE PROTEIN':
                    try:
                        chembl_def_dict[chembl_id].append(defin)
                    except:
                        chembl_def_dict[chembl_id] = [defin]
    return chembl_def_dict


def get_uniprot_chembl_target_id_mapping():
    uniprot_chembl_dict = dict()
    chembl_training_files_path = "{}/ChEMBL25/helper_files".format(training_files_path)
    with open("{}/{}".format(chembl_training_files_path, "chembl_uniprot_mapping.txt")) as f:
        for line in f:
            if not line.startswith("#") and line != "":
                line = line.split("\n")[0]
                u_id, chembl_id, defin, target_type = line.split("\t")
                if target_type == 'SINGLE PROTEIN':
                    try:
                        uniprot_chembl_dict[u_id].append(chembl_id)
                    except:
                        uniprot_chembl_dict[u_id] = [chembl_id]
    """
    for key in uniprot_chembl_dict.keys():
        if len(uniprot_chembl_dict[key])!=1:
            print(key, uniprot_chembl_dict[key])
    """
    return uniprot_chembl_dict


def get_numpy_target_dict_combined_feature_vectors(training_data_name, target_or_compound, feature_lst):
    print(feature_lst)
    target_matrix_max_value_dict = get_max_values_for_target_types(feature_lst)
    training_dataset_path = "{}/{}".format(training_files_path, training_data_name)
    comp_feature_vector_path = "{}/compound_feature_vectors".format(training_dataset_path)
    tar_feature_vector_path = "{}/target_feature_vectors".format(training_dataset_path)
    feat_vec_path = tar_feature_vector_path if target_or_compound == "target" else comp_feature_vector_path
    df_dti_data = pd.read_csv("{}/dti_datasets/comp_targ_affinity.csv".format(training_dataset_path), header=None)
    set_training_target_ids = set(df_dti_data.iloc[:,1])
    available_targets  =set ()
    df_combined_features = dict()
    count = 0
    with open("{}/{}.tsv".format(feat_vec_path, feature_lst[0])) as f:
        for line in f:
            line = line.split("\n")[0]
            line = line.split("\t")
            target_id = line[0]
            available_targets.add(target_id)


            if target_id!="target id" and target_id in set_training_target_ids:
                feat_vec = None
                prot_feature_dict = dict()
                # first channel represents the encoding
                prot_feature_dict[feature_lst[0]] = line[1:]

                for feature in feature_lst[1:]:
                    feature_matrix_fl = open(
                        "{}/target_feature_vectors/{}/{}.tsv".format(training_dataset_path, feature, target_id),
                        "r")
                    prot_feature_dict[feature] = feature_matrix_fl.read().split("\n")[0].split("\t")[1:]
                    feature_matrix_fl.close()

                if "500" in feature_lst[0]:
                    prot_all_channel_features = []


                    for feature in feature_lst:
                        prot_all_channel_features.append(
                            np.asarray(prot_feature_dict[feature], dtype=float)/target_matrix_max_value_dict[
                                feature.split("LEQ")[0]])

                    feat_vec = torch.tensor(
                        np.asarray(prot_all_channel_features, dtype=float).reshape(len(feature_lst), 500, 500)).type(
                        torch.FloatTensor)

                elif "1000" in feature_lst[0]:
                    prot_all_channel_features = []

                    for feature in feature_lst:
                        prot_all_channel_features.append(
                            np.asarray(prot_feature_dict[feature], dtype=float) / target_matrix_max_value_dict[
                                feature.split("LEQ")[0]])
                    feat_vec = torch.tensor(
                        np.asarray(prot_all_channel_features, dtype=float).reshape(len(feature_lst), 1000, 1000)).type(
                        torch.FloatTensor)
                else:
                    pass

                df_combined_features[target_id] = feat_vec
                count+=1
    # print(len(available_targets))
    return df_combined_features


def get_numpy_target_dict_combined_feature_vectors_single(training_data_name, target_id, target_or_compound, feature_lst):
    sorted(feature_lst)
    training_dataset_path = "{}/{}".format(training_files_path, training_data_name)
    comp_feature_vector_path = "{}/compound_feature_vectors".format(training_dataset_path)
    tar_feature_vector_path = "{}/target_feature_vectors".format(training_dataset_path)
    feat_vec_path = tar_feature_vector_path if target_or_compound == "target" else comp_feature_vector_path

    feat_vec_fl = open("{}/{}_normalized/{}.tsv".format(feat_vec_path, feature_lst[0], target_id), "r")
    line = feat_vec_fl.read().split("\n")[0]
    feat_vec_fl.close()
    line = line.split("\t")
    target_id = line[0]
    return torch.tensor(np.asarray([line[1:]], dtype=float).reshape(1, 500, 500)).type(torch.FloatTensor)


def get_compound_dict_combined_feature_vectors(training_data_name, feature_lst, fl_name=None):
    sorted(feature_lst)
    training_dataset_path = "{}/{}".format(training_files_path, training_data_name)
    comp_feature_vector_path = "{}/compound_feature_vectors".format(training_dataset_path)
    #tar_feature_vector_path = "{}/target_feature_vectors".format(training_dataset_path)
    #feat_vec_path = tar_feature_vector_path if target_or_compound == "target" else comp_feature_vector_path
    feat_vec_path = comp_feature_vector_path
    combined_features_dict = dict()
    count = 0
    feature_fl_path = "{}/{}_normalized.tsv".format(feat_vec_path, feature_lst[0]) if fl_name==None else "{}/{}".format(feat_vec_path, fl_name)
    with open(feature_fl_path) as f:
        for line in f:
            if count ==0:
                count+=1
            else:
                line = line.split("\n")[0]
                line = line.split("\t")
                target_id = line[0]
                feat_vec = line[1:]
                combined_features_dict[target_id] = torch.tensor(np.asarray(feat_vec, dtype=float)).type(torch.FloatTensor)
                count+=1
    return combined_features_dict


class CNNBioactivityDataset(Dataset):
    def __init__(self, training_data_name, comp_target_pair_dataset, compound_feature_list, target_feature_list, external_comp_feat_fl=None):
        self.training_data_name = training_data_name
        self.compound_feature_list = compound_feature_list
        self.target_feature_list = target_feature_list
        training_dataset_path = "{}/{}".format(training_files_path, training_data_name)
        comp_tar_training_dataset_path = "{}/dti_datasets".format(training_dataset_path)
        comp_target_pair_dataset_path = "{}/{}".format(comp_tar_training_dataset_path, comp_target_pair_dataset)

        self.dict_compound_features = get_compound_dict_combined_feature_vectors(training_data_name, compound_feature_list)
        self.dict_target_features = get_numpy_target_dict_combined_feature_vectors(training_data_name, "target", target_feature_list)
        self.training_dataset = pd.read_csv(comp_target_pair_dataset_path, header=None)
        self.num_of_train_test_val = len(self.training_dataset)
        # get_target_ids
        tar_ids = set(self.training_dataset.iloc[:, 1])
        #print(len(tar_ids))
        #print(self.training_dataset)
        #add test compound features
        if external_comp_feat_fl:
            test_rows = []
            test_compound_features_dict = get_compound_dict_combined_feature_vectors(training_data_name, compound_feature_list, external_comp_feat_fl)
            # print(test_compound_features_dict)
            for test_comp_id in test_compound_features_dict:
                #print(test_comp_id)
                self.dict_compound_features["test_{}".format(test_comp_id)] = test_compound_features_dict[test_comp_id]
                for tar_id in tar_ids:
                    test_rows.append(["test_{}".format(test_comp_id), tar_id, -1.0])
            df_test_rows = pd.DataFrame(test_rows)
            # print(df_test_rows)
            self.training_dataset = self.training_dataset.append(df_test_rows, ignore_index=True)

    def get_num_of_train_test_val(self):
        return self.num_of_train_test_val

    def __len__(self):
        return len(self.training_dataset)

    def __getitem__(self, idx):
        row = self.training_dataset.iloc[idx]
        comp_id, tar_id, biact_val = str(row[0]), str(row[1]), str(row[2])
        comp_feats = self.dict_compound_features[comp_id]
        tar_feats = self.dict_target_features[tar_id]
        # tar_feats = get_numpy_target_dict_combined_feature_vectors_single(self.training_data_name, tar_id, "target", self.target_feature_list)
        label = torch.tensor(float(biact_val)).type(torch.FloatTensor)
        return comp_feats, tar_feats, label, comp_id, tar_id


class CNNBioactivityTestDataset(Dataset):
    def __init__(self, training_data_name, comp_target_pair_dataset, compound_feature_list, target_feature_list, external_comp_feat_fl):
        self.training_data_name = training_data_name
        self.compound_feature_list = compound_feature_list
        self.target_feature_list = target_feature_list
        training_dataset_path = "{}/{}".format(training_files_path, training_data_name)
        comp_tar_training_dataset_path = "{}/dti_datasets".format(training_dataset_path)
        comp_target_pair_dataset_path = "{}/{}".format(comp_tar_training_dataset_path, comp_target_pair_dataset)

        self.dict_compound_features = dict()
        self.dict_target_features = get_numpy_target_dict_combined_feature_vectors(training_data_name, "target", target_feature_list)
        self.training_dataset = pd.read_csv(comp_target_pair_dataset_path, header=None)
        self.num_of_train_test_val = len(self.training_dataset)
        tar_ids = set(self.training_dataset.iloc[:, 1])

        test_rows = []
        test_compound_features_dict = get_compound_dict_combined_feature_vectors(training_data_name, compound_feature_list, external_comp_feat_fl)
        # print(test_compound_features_dict)
        for test_comp_id in test_compound_features_dict:
            #print(test_comp_id)
            self.dict_compound_features["test_{}".format(test_comp_id)] = test_compound_features_dict[test_comp_id]
            for tar_id in tar_ids:
                test_rows.append(["test_{}".format(test_comp_id), tar_id, -1.0])
        df_test_rows = pd.DataFrame(test_rows)
        # print(df_test_rows)
        self.test_dataset = df_test_rows

    def __len__(self):
        return len(self.test_dataset)

    def __getitem__(self, idx):
        row = self.test_dataset.iloc[idx]
        comp_id, tar_id, biact_val = str(row[0]), str(row[1]), str(row[2])
        comp_feats = self.dict_compound_features[comp_id]
        tar_feats = self.dict_target_features[tar_id]
        # tar_feats = get_numpy_target_dict_combined_feature_vectors_single(self.training_data_name, tar_id, "target", self.target_feature_list)
        label = torch.tensor(float(biact_val)).type(torch.FloatTensor)
        return comp_feats, tar_feats, label, comp_id, tar_id

def get_cnn_test_data_loader(training_data_name, comp_feature_list, tar_feature_list, external_comp_feat_fl, batch_size=32):

    bioactivity_dataset = CNNBioactivityTestDataset(training_data_name, compound_target_pair_dataset, comp_feature_list, tar_feature_list, external_comp_feat_fl)

    external_test_indices = list(range(len(bioactivity_dataset)))
    print(external_test_indices)
    external_test_sampler = SubsetRandomSampler(external_test_indices)
    external_test_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                       sampler=external_test_sampler)

    return external_test_loader


def get_test_list_target_dict_combined_feature_vectors(training_data_name, target_or_compound, feature_lst):
    sorted(feature_lst)
    training_dataset_path = "{}/{}".format(training_files_path, training_data_name)
    comp_feature_vector_path = "{}/compound_feature_vectors".format(training_dataset_path)
    tar_feature_vector_path = "{}/target_feature_vectors".format(training_dataset_path)
    feat_vec_path = tar_feature_vector_path if target_or_compound == "target" else comp_feature_vector_path
    common_column = "target id" if target_or_compound=="target" else "compound id"
    df_combined_features = dict()
    count = 0
    with open("{}/test_{}_normalized.tsv".format(feat_vec_path, feature_lst[0])) as f:
        for line in f:
            line = line.split("\n")[0]
            line = line.split("\t")
            target_id = line[0]
            feat_vec = line[1:]
            df_combined_features[target_id] = torch.tensor(np.asarray(feat_vec, dtype=float)).type(torch.FloatTensor)
            count+=1
    return df_combined_features


def get_cnn_test_val_folds_train_data_loader(training_data_name, comp_feature_list, tar_feature_list, batch_size=32, external_comp_feat_fl=None):
    import json
    training_dataset_path = "{}/{}".format(training_files_path, training_data_name)
    folds_path = "{}/data/folds".format(training_dataset_path)

    folds = json.load(open("{}/train_fold_setting1.txt".format(folds_path)))
    test_indices = json.load(open("{}/test_fold_setting1.txt".format(folds_path)))

    bioactivity_dataset = CNNBioactivityDataset(training_data_name, compound_target_pair_dataset, comp_feature_list, tar_feature_list, external_comp_feat_fl=external_comp_feat_fl)
    num_of_train_test_val = bioactivity_dataset.get_num_of_train_test_val()
    num_of_all_data_points = len(bioactivity_dataset)

    loader_fold_dict = dict()
    for fold_id in range(len(folds)):
        folds_id_list = list(range(len(folds)))
        val_indices = folds[fold_id]
        folds_id_list.remove(fold_id)
        train_indices  = []
        for tr_fold_in in folds_id_list:
            train_indices.extend(folds[tr_fold_in])
        train_indices = train_indices#[:10]
        val_indices = val_indices#[:10]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                   sampler=train_sampler)

        valid_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                   sampler=valid_sampler)

        loader_fold_dict[fold_id] = [train_loader, valid_loader]

    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                               sampler=test_sampler)
    external_test_loader = None
    if external_comp_feat_fl:
        external_test_indices = list(range(num_of_train_test_val,num_of_all_data_points))
        print(external_test_indices)
        external_test_sampler = SubsetRandomSampler(external_test_indices)
        external_test_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                            sampler=external_test_sampler)
    return loader_fold_dict, test_loader, external_test_loader


#get_cnn_test_val_folds_train_data_loader("Davis_Filtered", ["ecfp4"], ["sequencematrix500"], external_comp_feat_fl="aacrtest_ecfp4_normalized.tsv")


def get_cnn_test_val_folds_train_data_loader(training_data_name, comp_feature_list, tar_feature_list, batch_size=32, external_comp_feat_fl=None):
    import json
    training_dataset_path = "{}/{}".format(training_files_path, training_data_name)
    folds_path = "{}/data/folds".format(training_dataset_path)

    folds = json.load(open("{}/train_fold_setting1.txt".format(folds_path)))
    test_indices = json.load(open("{}/test_fold_setting1.txt".format(folds_path)))

    bioactivity_dataset = CNNBioactivityDataset(training_data_name, compound_target_pair_dataset, comp_feature_list, tar_feature_list, external_comp_feat_fl=external_comp_feat_fl)
    num_of_train_test_val = bioactivity_dataset.get_num_of_train_test_val()
    num_of_all_data_points = len(bioactivity_dataset)

    loader_fold_dict = dict()
    for fold_id in range(len(folds)):
        folds_id_list = list(range(len(folds)))
        val_indices = folds[fold_id]
        folds_id_list.remove(fold_id)
        train_indices  = []
        for tr_fold_in in folds_id_list:
            train_indices.extend(folds[tr_fold_in])
        train_indices = train_indices#[:10]
        val_indices = val_indices#[:10]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                   sampler=train_sampler)

        valid_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                   sampler=valid_sampler)

        loader_fold_dict[fold_id] = [train_loader, valid_loader]

    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                               sampler=test_sampler)
    external_test_loader = None
    if external_comp_feat_fl:
        external_test_indices = list(range(num_of_train_test_val,num_of_all_data_points))
        print(external_test_indices)
        external_test_sampler = SubsetRandomSampler(external_test_indices)
        external_test_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                            sampler=external_test_sampler)
    return loader_fold_dict, test_loader, external_test_loader


"""
class CNNBioactivityTestDataset(Dataset):
    def __init__(self, test_data_name, comp_target_pair_dataset, compound_feature_list, target_feature_list):
        self.training_data_name = test_data_name
        self.compound_feature_list = compound_feature_list
        self.target_feature_list = target_feature_list
        training_dataset_path = "{}/{}".format(training_files_path, test_data_name)
        comp_tar_training_dataset_path = "{}/dti_datasets".format(training_dataset_path)
        comp_target_pair_dataset_path = "{}/{}".format(comp_tar_training_dataset_path, comp_target_pair_dataset)

        self.dict_compound_features = get_test_list_target_dict_combined_feature_vectors(test_data_name, "compound", compound_feature_list)
        self.dict_target_features = get_numpy_target_dict_combined_feature_vectors(test_data_name, "target", target_feature_list)
        self.training_dataset = pd.read_csv(comp_target_pair_dataset_path, header=None)
    def __len__(self):
        return len(self.training_dataset)

    def __getitem__(self, idx):
        row = self.training_dataset.iloc[idx]
        comp_id, tar_id, biact_val = str(row[0]), str(row[1]), str(row[2])
        comp_feats = self.dict_compound_features[comp_id]
        tar_feats = self.dict_target_features[tar_id]
        # tar_feats = get_numpy_target_dict_combined_feature_vectors_single(self.training_data_name, tar_id, "target", self.target_feature_list)
        label = torch.tensor(float(biact_val)).type(torch.FloatTensor)
        return comp_feats, tar_feats, label, comp_id, tar_id

def get_cnn_test_data_loader(training_data_name, comp_feature_list, tar_feature_list, batch_size=32):
    import numpy as np
    import json
    compound_target_pair_dataset = "test_comp_targ_affinity.csv"


    test = list(range(1384))

    bioactivity_dataset = CNNBioactivityTestDataset(training_data_name, compound_target_pair_dataset, comp_feature_list, tar_feature_list)

    test_sampler = SubsetRandomSampler(test)
    test_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                   sampler=test_sampler)

    return test_loader
"""
def get_cnn_train_test_full_training_data_loader(training_data_name, comp_feature_list, tar_feature_list, batch_size=32, train_val_test=False, external_comp_feat_fl=None):
    import json

    training_dataset_path = "{}/{}".format(training_files_path, training_data_name)
    folds_path = "{}/data/folds".format(training_dataset_path)

    folds = json.load(open("{}/train_fold_setting1.txt".format(folds_path)))
    test = json.load(open("{}/test_fold_setting1.txt".format(folds_path)))

    bioactivity_dataset = CNNBioactivityDataset(training_data_name, compound_target_pair_dataset, comp_feature_list, tar_feature_list, external_comp_feat_fl=external_comp_feat_fl)

    num_of_train_test_val = bioactivity_dataset.get_num_of_train_test_val()
    num_of_all_data_points = len(bioactivity_dataset)
    print(num_of_train_test_val, num_of_all_data_points)

    train_indices = []
    validation_indices = []

    if train_val_test:
        for fold in folds[:-1]:
            train_indices.extend(fold)
        validation_indices = folds[-1]
    else:
        for fold_id in range(len(folds)):
            train_indices.extend(folds[fold_id])

    train_sampler = SubsetRandomSampler(train_indices)

    train_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                               sampler=train_sampler)

    validation_sampler, validation_loader = None, None
    if train_val_test:
        validation_sampler = SubsetRandomSampler(validation_indices)
        validation_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                   sampler=validation_sampler)


    test_sampler = SubsetRandomSampler(test)
    test_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                   sampler=test_sampler)

    external_test_loader = None
    if external_comp_feat_fl:
        external_test_indices = list(range(num_of_train_test_val, num_of_all_data_points))
        # print(external_test_indices)
        external_test_sampler = SubsetRandomSampler(external_test_indices)
        external_test_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                           sampler=external_test_sampler)

    if train_val_test:
        return train_loader, validation_loader, test_loader, external_test_loader

    return train_loader, test_loader, external_test_loader




