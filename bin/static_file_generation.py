import subprocess
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import itertools

def get_aa_list():
    aa_list = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    return aa_list

def remove_nonstandard_aas(prot_seq):
    aa_list = get_aa_list()
    prot_seq_list = [aa for aa in prot_seq if aa in aa_list]
    prot_seq = ''.join(prot_seq_list)
    return prot_seq

def get_all_aa_word_list(word_size):
    aa_list = get_aa_list()
    # all_n_gram_list = list(itertools.permutations(aa_list, word_size))
    all_n_gram_list = list(itertools.product(aa_list, repeat=word_size))
    all_n_gram_list = [''.join(n_gram_tuple) for n_gram_tuple in all_n_gram_list]
    return all_n_gram_list

def get_aa_match_encodings():
    all_aa_matches = get_all_aa_word_list(2)
    aa_match_encoding_dict = dict()
    encod_int = 1
    for aa_pair in all_aa_matches:
        if aa_pair not in aa_match_encoding_dict.keys():
            aa_match_encoding_dict[aa_pair] = encod_int
            aa_match_encoding_dict[aa_pair[::-1]] = encod_int
            encod_int += 1
    return aa_match_encoding_dict

def get_prot_id_seq_dict_from_fasta_fl(fasta_fl_path):
    prot_id_seq_dict = dict()

    prot_id = ""
    with open("{}".format(fasta_fl_path)) as f:
        for line in f:
            line = line.split("\n")[0]
            if line.startswith(">"):
                prot_id = line.split("|")[1]
                prot_id_seq_dict[prot_id] = ""
            else:
                prot_id_seq_dict[prot_id] = prot_id_seq_dict[prot_id] + line

    return prot_id_seq_dict

def get_sequence_matrix(seq, size, aaindex_enconding=None):
    aa_match_encoding_dict = None
    if aaindex_enconding==None:
        aa_match_encoding_dict = get_aa_match_encodings()
    elif aaindex_enconding=="ZHAC000103":
        aa_match_encoding_dict = get_aa_match_encodings_generic(aaindex_enconding, full_matrix=True)
    else:
        aa_match_encoding_dict = get_aa_match_encodings_generic(aaindex_enconding)
    # print(aa_match_encoding_dict)

    seq = remove_nonstandard_aas(seq)
    lst = []
    for i in range(len(seq)):
        lst.append([])
        for j in range(len(seq)):
            lst[-1].append(aa_match_encoding_dict[seq[i] + seq[j]])

    torch_arr = torch.from_numpy(np.asarray(lst))
    size_of_tensor = torch_arr.shape[0]
    # print(torch_list)
    # print(torch_list.shape[0])
    if size_of_tensor < size:
        padding_size = int((size - size_of_tensor) / 2)
        m = nn.ZeroPad2d(padding_size)
        if size_of_tensor % 2 != 0:
            m = nn.ZeroPad2d((padding_size, padding_size + 1, padding_size, padding_size + 1))
        torch_arr = m(torch_arr)
    else:
        torch_arr = torch_arr[:size, :size]

    # print(torch_arr.shape)
    return torch_arr


def get_aa_match_encodings_generic(aaindex_enconding, full_matrix=False):
    aa_list = get_aa_list()
    encoding_fl = open("../training_files/encodings/{}".format(aaindex_enconding))
    lst_encoding_fl = encoding_fl.read().split("\n")
    encoding_fl.close()
    aa_match_encoding_dict = dict()
    starting_ind = -1
    for row_ind in range(len(lst_encoding_fl)-1):
        str_line = lst_encoding_fl[row_ind]
        if str_line.startswith("M rows"):
            starting_ind = row_ind + 1

        if  not str_line.startswith("//") and starting_ind != -1 and row_ind >= starting_ind:
            row_aa_ind = row_ind - starting_ind
            str_line = str_line.split(" ")

            while "" in str_line:
                str_line.remove("")

            # print(len(str_line))
            for col_ind in range(len(str_line)):
                # print(str_line)
                # print(row_aa_ind, col_ind, aa_list[row_aa_ind], aa_list[col_ind], str_line[col_ind] )
                if not full_matrix:
                    aa_match_encoding_dict["{}{}".format(aa_list[row_aa_ind], aa_list[col_ind])] = round(float(str_line[col_ind]),3)
                    aa_match_encoding_dict["{}{}".format(aa_list[col_ind], aa_list[row_aa_ind])] = round(float(str_line[col_ind]),3)
                else:
                    aa_match_encoding_dict["{}{}".format(aa_list[row_aa_ind], aa_list[col_ind])] = round(
                        float(str_line[col_ind]), 3)
    #print(aa_match_encoding_dict)
    return aa_match_encoding_dict

def save_separate_flattened_sequence_matrices(dataset_name, size, aaindex_enconding=None):
    print(dataset_name, size, aaindex_enconding)
    target_fl_name = "targets.fasta"
    fasta_fl_path = "../training_files/{}/helper_files/{}".format(dataset_name, target_fl_name)
    prot_id_seq_dict = get_prot_id_seq_dict_from_fasta_fl(fasta_fl_path)

    feature_name = ""
    if aaindex_enconding:
        feature_name = "{}LEQ{}".format(aaindex_enconding.split(".")[0], size)
    else:
        feature_name = "sequencematrix"

    str_header = "target id\t" + "\t".join([str(num) for num in list(range(size * size))])
    count = 0
    output_fl_name = ""
    if feature_name == "sequencematrix":
        output_fl_name = "{}{}.tsv".format(feature_name, size)
        output_fl = open("../training_files/{}/target_feature_vectors/{}".format(dataset_name, output_fl_name), "w")
        output_fl.write(str_header + "\n")
        for prot_id, seq in prot_id_seq_dict.items():
            count += 1
            if count % 100 == 0:
                print(count)
            seq_torch_matrix = None
            if aaindex_enconding == None:
                seq_torch_matrix = get_sequence_matrix(seq, size)
            else:
                seq_torch_matrix = get_sequence_matrix(seq, size, aaindex_enconding)
            flattened_seq_matrix_arr = np.array(seq_torch_matrix.contiguous().view(-1))
            output_fl.write(prot_id + "\t" + "\t".join([str(val) for val in flattened_seq_matrix_arr]) + "\n")

        output_fl.close()
    else:
        # feature_name = "blosum62LEQ500"
        print(feature_name)

        output_folder_name = "{}".format(feature_name, size)
        output_path = "../training_files/{}/target_feature_vectors/{}".format(dataset_name, output_folder_name)
        subprocess.call("mkdir {}".format(output_path), shell=True)
        for prot_id, seq in prot_id_seq_dict.items():
            count += 1
            if count % 100 == 0:
                print(count)
            seq_torch_matrix = None
            if aaindex_enconding == None:
                seq_torch_matrix = get_sequence_matrix(seq, size)
            else:
                seq_torch_matrix = get_sequence_matrix(seq, size, aaindex_enconding)
            flattened_seq_matrix_arr = np.array(seq_torch_matrix.contiguous().view(-1))
            output_fl = open("{}/{}.tsv".format(output_path, prot_id), "w")
            output_fl.write(prot_id + "\t" + "\t".join([str(val) for val in flattened_seq_matrix_arr]))
            output_fl.close()