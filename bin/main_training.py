import argparse
from train_mdeepred import five_fold_training, train_val_test_training


parser = argparse.ArgumentParser(description='MDeePred arguments')
parser.add_argument(
    '--chln',
    type=str,
    default="512_512",
    metavar='CHLN',
    help='number of neurons in compound hidden layers (default: 512_512)')

parser.add_argument(
    #'--target-layer-neurons-after-flattened',
    '--tlnaf',
    type=int,
    default=512,
    metavar='TFFLAF',
    help='number of neurons after flattening target conv layers (default: 512)')
parser.add_argument(
    '--lhln',
    type=str,
    default="256_256",
    metavar='LHLN',
    help='number of neurons in last two hidden layers before output layer (default: 256_256)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    metavar='LR',
    help='learning rate (default: 0.001)')

parser.add_argument(
    # '--batch-size',
    '--bs',
    type=int,
    default=32,
    metavar='BS',
    help='batch size (default: 32)')
parser.add_argument(
    # '--training-data',
    '--td',
    type=str,
    default="Davis_Filtered",
    metavar='TD',
    help='the name of the training dataset (default: Davis_Filtered)')

parser.add_argument(
    # '--compound-features',
    '--cf',
    type=str,
    default="ecfp4",
    metavar='CF',
    help='compound features separated by underscore character (default: ecfp4)')

parser.add_argument(
    # '--target-features',
    '--tf',
    type=str,
    default="sequencematrix500",
    metavar='TF',
    help='target features separated by underscore character (default: sequencematrix500)')

parser.add_argument(
    '--setting',
    type=int,
    default=1,
    metavar='TVT',
    help='Determines if data is divided into train-validation-test (default: 1)')

parser.add_argument(
    '--dropout',
    type=float,
    default=0.25,
    metavar='DO',
    help='dropout rate (default: 0.25)')

parser.add_argument(
    '--en',
    type=str,
    default="my_experiments",
    metavar='EN',
    help='the name of the experiment (default: my_experiment)')


parser.add_argument(
    '--model',
    type=str,
    default="CompFCNNTarCNNModuleInception",
    metavar='mn',
    help='model name (default: CompFCNNTarCNNModuleInception)')



if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    comp_hidden_layer_neurons = [int(num) for num in args.chln.split("_")]
    last_2_hidden_layer_list = [int(num) for num in args.lhln.split("_")]

    if args.setting == 1:
        five_fold_training(args.td, (args.cf).split("_"), (args.tf).split("_"), comp_hidden_layer_neurons,
                           args.tlnaf, last_2_hidden_layer_list[0], last_2_hidden_layer_list[1], args.lr,
                           args.bs, args.model, args.dropout, args.en)

    # This setting is for both train_validation and test split and time-split
    elif args.setting == 2:
        train_val_test_training(args.td, (args.cf).split("_"), (args.tf).split("_"), comp_hidden_layer_neurons,
                                args.tlnaf, last_2_hidden_layer_list[0], last_2_hidden_layer_list[1], args.lr,
                                args.bs, args.model, args.dropout, args.en)
    else:
        pass
