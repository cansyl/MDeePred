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
    metavar='set',
    help='Determines the setting (1: n_fold, 2:train_val_test) (default: 1)')

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
parser.add_argument(
    '--epoch',
    type=int,
    default=200,
    metavar='EPC',
    help='Number of epochs (default: 200)')
parser.add_argument(
    '--fold_num',
    type=int,
    default=None,
    metavar='fn',
    help='Determines the fold number to train this is for independent fold training (default: None)')
parser.add_argument(
    '--train_val_test',
    type=int,
    default=1,
    metavar='TVT',
    help='Determines if train-test or train_val_test (default: 1)')
parser.add_argument(
    '--ext_test_feat_vec',
    type=str,
    default=None,
    metavar='ETFV',
    help='The name of the external test feature vector file (default: None)')




if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    comp_hidden_layer_neurons = [int(num) for num in args.chln.split("_")]
    last_2_hidden_layer_list = [int(num) for num in args.lhln.split("_")]

    if args.setting == 1:
        five_fold_training(args.td, (args.cf).split("_"), (args.tf).split("_"), comp_hidden_layer_neurons,
                           args.tlnaf, last_2_hidden_layer_list[0], last_2_hidden_layer_list[1], args.lr,
                           args.bs, args.model, args.dropout, args.en, args.epoch, args.fold_num, args.ext_test_feat_vec)

    # This setting is for both train_validation and test split and time-split
    elif args.setting == 2:
        print("Setting 2", args.train_val_test)
        train_val_test_training(args.td, (args.cf).split("_"), (args.tf).split("_"), comp_hidden_layer_neurons,
                                args.tlnaf, last_2_hidden_layer_list[0], last_2_hidden_layer_list[1], args.lr,
                                args.bs, args.model, args.dropout, args.en, args.epoch, args.train_val_test, args.ext_test_feat_vec)
        # python main_training.py --td kinome --args.setting 2 --cf ecfp4 --tf sequencematrix1000_ZHAC000103LEQ1000_GRAR740104LEQ1000_SIMK990101LEQ1000_blosum62LEQ1000 --chln 1024_1024
        # --tlnaf 256 --lhln 512_256 --lr 0.0001 --bs 32 --model CompFCNNTarCNNModuleInception --dropout 0.25 --final_kinome --epoch 200 --train_val_test True
        #
    else:
        pass
