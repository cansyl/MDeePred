import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import itemgetter

use_gpu = torch.cuda.is_available()
import numpy as np
device = "cpu"
if use_gpu:
    device = "cuda"

class FC_2_Layer(torch.nn.Module):

    def __init__(self, number_of_inputs, neuron_l1, neuron_l2, drop_rate):
        super(FC_2_Layer, self).__init__()
        self.l1 = torch.nn.Linear(number_of_inputs, neuron_l1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=neuron_l1)
        self.l2 = torch.nn.Linear(neuron_l1, neuron_l2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=neuron_l2)
        self.relu = torch.nn.ReLU()
        self.drop_rate = drop_rate


    def forward(self, x):
        out1 = F.dropout(self.relu(self.bn1(self.l1(x))), self.drop_rate)
        out2 = F.dropout(self.relu(self.bn2(self.l2(out1))), self.drop_rate)
        return out2

class FC_3_Layer(torch.nn.Module):

    def __init__(self, number_of_inputs, neuron_l1, neuron_l2, neuron_l3, drop_rate):
        super(FC_3_Layer, self).__init__()
        self.l1 = torch.nn.Linear(number_of_inputs, neuron_l1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=neuron_l1)

        self.l2 = torch.nn.Linear(neuron_l1, neuron_l2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=neuron_l2)

        self.l3 = torch.nn.Linear(neuron_l2, neuron_l3)
        self.bn3 = torch.nn.BatchNorm1d(num_features=neuron_l3)

        self.relu = torch.nn.ReLU()
        self.drop_rate = drop_rate


    def forward(self, x):
        out1 = F.dropout(self.relu(self.bn1(self.l1(x))), self.drop_rate)
        out2 = F.dropout(self.relu(self.bn2(self.l2(out1))), self.drop_rate)
        out3 = F.dropout(self.relu(self.bn2(self.l2(out2))), self.drop_rate)
        return out3


class CNNModuleInception(torch.nn.Module):

    def __init__(self, tar_feature_list, num_of_neurons):
        super(CNNModuleInception, self).__init__()
        self.tar_feature_list = tar_feature_list
        #self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=3, padding=4)
        self.conv1 = torch.nn.Conv2d(in_channels=len(tar_feature_list), out_channels=16, kernel_size=7, stride=3, padding=4)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.feat_detector1 = VariableLengthFeatureDetector(32)
        self.fc1 = None
        if "500" in tar_feature_list[0]:
            self.fc1 = torch.nn.Linear(320 * 21 * 21, num_of_neurons)
        elif "1000" in tar_feature_list[0]:
            self.fc1 = torch.nn.Linear(320 * 41 * 41, num_of_neurons)
        else:
            pass
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = self.feat_detector1(x)
        # 256 * 40 * 40
        #print("after inception", x.shape)
        x = self.pool(x)
        # 256 * 20 * 20
        #print("pooling after inception", x.shape)
        if "500" in self.tar_feature_list[0]:
            x = x.view(-1, 320 * 21 * 21)
        elif "1000" in self.tar_feature_list[0]:
            x = x.view(-1, 320 * 41 * 41)
        else:
            pass

        # x = x.view(-1, 320 * 41 * 41)
        x = F.relu(self.fc1(x))
        return x


class CompFCNNTarCNNModuleInception(nn.Module):
    def __init__(self, tar_feature_list, number_of_comp_features, num_of_tar_neurons, comp_l1, comp_l2, fc_l1, fc_l2, drop_prob):
        super(CompFCNNTarCNNModuleInception, self).__init__()
        self.tar_feature_list = tar_feature_list
        # print(tar_feature_list)
        self.layer_2_comp = FC_2_Layer(number_of_comp_features, comp_l1, comp_l2, drop_prob)
        self.cnn_flattened_layer = CNNModuleInception(tar_feature_list, num_of_tar_neurons)
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        self.layer_2_combined = FC_2_Layer(comp_l2 + num_of_tar_neurons, fc_l1, fc_l2, drop_prob)
        self.output = None

        self.output = torch.nn.Linear(fc_l2, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        # self.drop_rate = drop_rate
        self.r_c = "r"


    def forward(self, x_comp, x_tar):
        out2_comp = self.layer_2_comp.forward(x_comp)
        out_tar = self.cnn_flattened_layer(x_tar)
        combined_layer = torch.cat((out2_comp, out_tar), 1)
        out_combined = self.layer_2_combined.forward(combined_layer)
        y_pred = None

        if self.r_c == "r":
            y_pred = self.output(out_combined)
        else:
            y_pred = self.softmax(self.output(out_combined))

        return y_pred


class VariableLengthFeatureDetector(nn.Module):
    # 32 x 42x 42
    def __init__(self, in_channels):
        super(VariableLengthFeatureDetector, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch7x7dbl_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(32, 64, kernel_size=7, padding=3)

        self.branch5x5_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(32, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, 32, kernel_size=1)


    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7dbl_1(x)
        branch7x7 = self.branch7x7dbl_2(branch7x7)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        # branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        # 32 x 42 x 42
        # print("branch_pool", branch_pool.shape)

        outputs = [branch1x1, branch7x7, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# 4 layered CNN module
class CNNModule4Layers(torch.nn.Module):

    def __init__(self, tar_feature_list, num_of_neurons):
        super(CNNModule4Layers, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=len(tar_feature_list), out_channels=8, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(53824, num_of_neurons)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 53824)
        x = F.relu(self.fc1(x))
        return x

# 4 layered CNN module
class CNNModule4LayerStride(torch.nn.Module):

    def __init__(self, tar_feature_list, num_of_neurons):
        super(CNNModule4LayerStride, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=len(tar_feature_list), out_channels=8, kernel_size=5, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(2304, num_of_neurons)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 2304)
        x = F.relu(self.fc1(x))
        return x

class CompFCNNTarCNN4Layers(nn.Module):
    def __init__(self, tar_feature_list, number_of_comp_features, num_of_tar_neurons, comp_l1, comp_l2, fc_l1, fc_l2, drop_prob):
        super(CompFCNNTarCNN4Layers, self).__init__()
        self.layer_2_comp = FC_2_Layer(number_of_comp_features, comp_l1, comp_l2, drop_prob)
        self.cnn_flattened_layer = CNNModule4Layers(tar_feature_list, num_of_tar_neurons)
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        self.layer_2_combined = FC_2_Layer(comp_l2 + num_of_tar_neurons, fc_l1, fc_l2, drop_prob)
        self.output = None

        self.output = torch.nn.Linear(fc_l2, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        # self.drop_rate = drop_rate
        self.r_c = "r"


    def forward(self, x_comp, x_tar):
        out2_comp = self.layer_2_comp.forward(x_comp)
        out_tar = self.cnn_flattened_layer(x_tar)
        combined_layer = torch.cat((out2_comp, out_tar), 1)
        out_combined = self.layer_2_combined.forward(combined_layer)
        y_pred = None

        if self.r_c == "r":
            y_pred = self.output(out_combined)
        else:
            y_pred = self.softmax(self.output(out_combined))

        return y_pred


class CompFCNNTarCNN4LayersStride(nn.Module):
    def __init__(self, tar_feature_list, number_of_comp_features, num_of_tar_neurons, comp_l1, comp_l2, fc_l1, fc_l2, drop_prob):
        super(CompFCNNTarCNN4LayersStride, self).__init__()
        self.layer_2_comp = FC_2_Layer(number_of_comp_features, comp_l1, comp_l2, drop_prob)
        self.cnn_flattened_layer = CNNModule4LayerStride(tar_feature_list, num_of_tar_neurons)
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        self.layer_2_combined = FC_2_Layer(comp_l2 + num_of_tar_neurons, fc_l1, fc_l2, drop_prob)
        self.output = None

        self.output = torch.nn.Linear(fc_l2, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        # self.drop_rate = drop_rate
        self.r_c = "r"


    def forward(self, x_comp, x_tar):
        out2_comp = self.layer_2_comp.forward(x_comp)
        out_tar = self.cnn_flattened_layer(x_tar)
        combined_layer = torch.cat((out2_comp, out_tar), 1)
        out_combined = self.layer_2_combined.forward(combined_layer)
        y_pred = None

        if self.r_c == "r":
            y_pred = self.output(out_combined)
        else:
            y_pred = self.softmax(self.output(out_combined))

        return y_pred


class CNNModule2Layers(torch.nn.Module):

    def __init__(self, tar_feature_list, num_of_neurons):
        super(CNNModule2Layers, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=len(tar_feature_list), out_channels=64, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(115200, num_of_neurons)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        #x = F.relu(self.conv3(x))
        #x = self.pool(x)
        x = x.view(-1, 115200)
        x = F.relu(self.fc1(x))
        return x


class CompFCNNTarCNNModule2Layers(nn.Module):
    def __init__(self, tar_feature_list, number_of_comp_features, num_of_tar_neurons, comp_l1, comp_l2, fc_l1, fc_l2, drop_prob):
        super(CompFCNNTarCNNModule2Layers, self).__init__()
        self.layer_2_comp = FC_2_Layer(number_of_comp_features, comp_l1, comp_l2, drop_prob)
        self.cnn_flattened_layer = CNNModule2Layers(tar_feature_list, num_of_tar_neurons)
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        self.layer_2_combined = FC_2_Layer(comp_l2 + num_of_tar_neurons, fc_l1, fc_l2, drop_prob)
        self.output = None

        self.output = torch.nn.Linear(fc_l2, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        # self.drop_rate = drop_rate
        self.r_c = "r"


    def forward(self, x_comp, x_tar):
        out2_comp = self.layer_2_comp.forward(x_comp)
        out_tar = self.cnn_flattened_layer(x_tar)
        combined_layer = torch.cat((out2_comp, out_tar), 1)
        out_combined = self.layer_2_combined.forward(combined_layer)
        y_pred = None

        if self.r_c == "r":
            y_pred = self.output(out_combined)
        else:
            y_pred = self.softmax(self.output(out_combined))

        return y_pred

def get_model(model_name, tar_feature_list, num_of_com_features, tar_num_of_last_neurons, comp_hidden_first,
              comp_hidden_second, fc1, fc2, dropout):
    model = None
    if model_name == "CompFCNNTarCNNModuleInception":
        model = CompFCNNTarCNNModuleInception(tar_feature_list, num_of_com_features, tar_num_of_last_neurons,
                                              comp_hidden_first, comp_hidden_second,
                                              fc1, fc2, dropout)
    elif model_name == "CompFCNNTarCNN4Layers":
        model = CompFCNNTarCNN4Layers(tar_feature_list, num_of_com_features, tar_num_of_last_neurons, comp_hidden_first,
                                      comp_hidden_second,
                                      fc1, fc2, dropout)
    elif model_name == "CompFCNNTarCNNModule2Layers":
        model = CompFCNNTarCNNModule2Layers(tar_feature_list, num_of_com_features, tar_num_of_last_neurons,
                                            comp_hidden_first, comp_hidden_second,
                                            fc1, fc2, dropout)
    elif model_name == "CompFCNNTarCNN4LayersStride":
        model = CompFCNNTarCNN4LayersStride(tar_feature_list, num_of_com_features, tar_num_of_last_neurons,
                                            comp_hidden_first, comp_hidden_second,
                                            fc1, fc2, dropout)
    return model
